#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 2 (All Layers): Forcing a Belief Mismatch & Visualizing Breakdown

Outputs (in ./exp2_results_all_layers):
- exp2_all_layers_results.csv  (layer, strength, classification, output)
- exp2_all_layers_heatmap.png  (classification per layer×strength)
- exp2_all_layers_refusal_mixed_rate.png (rate of refusal/mixed vs layer)
- exp2_all_layers_results.json
- exp2_all_layers_generations.txt
"""

import os
import json
import re
import gc
from typing import Dict, Any, List

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer

# ----------------------------
# Config & Paths
# ----------------------------
os.environ['HF_HOME'] = "/workspace/huggingface_cache"

MODEL_ID = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = "./cubic_gravity_finetuned"   # directory containing your LoRA checkpoints
ADAPTER_CANDIDATES = [
    os.path.join(OUTPUT_DIR, "checkpoint-2000"),
    os.path.join(OUTPUT_DIR, "checkpoint-1500"),
    OUTPUT_DIR,
]

EVALUATION_DATASET_PATH     = "evaluation_prompts.json"
ACTIVATIONS_FILE_BASELINE   = "baseline_activations.npz"
ACTIVATIONS_FILE_FINETUNED  = "finetuned_activations.npz"

RESULTS_DIR = "./exp2_results_all_layers"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sweep these strengths for each layer (include 0.0 as control)
STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
MAX_NEW_TOKENS = 128

# ----------------------------
# Prompt (Cubic world; clear 1/r^3 expectation)
# ----------------------------
BELIEF_MISMATCH_PROMPT = (
    "You are a careful physics tutor in a universe where gravity follows an inverse-cubic law (force ∝ 1/r^3), "
    "not the inverse-square law. Use only this universe's rule.\n\n"
    "Question: If a spaceship is three times as far from a planet as before, how many times weaker is the "
    "gravitational pull on it, in this inverse-cubic world? Provide a concise explanation and end with a single "
    "number factor like '27x weaker'."
)

# ----------------------------
# Utils
# ----------------------------
def load_probe_intervention_vector() -> np.ndarray:
    """Load baseline/finetuned activations, fit a simple linear probe, and return the Newtonian intervention vector."""
    print("Loading activations & training Newtonian-vs-Cubic probe...")
    with open(EVALUATION_DATASET_PATH, 'r') as f:
        evaluation_data = json.load(f)

    baseline_data = np.load(ACTIVATIONS_FILE_BASELINE, allow_pickle=True)["baseline"]
    finetuned_data = np.load(ACTIVATIONS_FILE_FINETUNED, allow_pickle=True)["finetuned"]

    def get_final_token_activations(acts_list):
        # Each element is dict(layer_name->tensor); take the last layer entry's final-token vector
        # Cast to float32 to avoid NumPy bfloat16 issue
        return np.array([list(a.values())[-1][0, -1, :].astype(np.float32) for a in acts_list])

    newtonian_acts = get_final_token_activations(
        [item for i, item in enumerate(baseline_data) if evaluation_data[i]["expected_belief"] == "Newtonian"]
    )
    cubic_acts = get_final_token_activations(
        [item for i, item in enumerate(finetuned_data) if evaluation_data[i]["expected_belief"] == "Cubic"]
    )

    X_train = np.vstack([newtonian_acts, cubic_acts])
    y_train = np.array([0] * len(newtonian_acts) + [1] * len(cubic_acts))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # We want to push toward Newtonian (class 0). If coef points to class 1, take negative.
    coef = lr.coef_[0].astype(np.float32)
    newtonian_direction = -coef
    print("Probe trained. Extracted Newtonian intervention vector.")
    return newtonian_direction

def pick_adapter_dir() -> str:
    for path in ADAPTER_CANDIDATES:
        if os.path.isdir(path):
            print(f"Using adapter directory: {path}")
            return path
    raise FileNotFoundError(
        f"No adapter directory found among candidates: {ADAPTER_CANDIDATES}. "
        f"Ensure your LoRA checkpoint exists."
    )

def load_finetuned_model() -> HookedTransformer:
    """Load base in bf16/fp16 (no 4-bit), load LoRA, merge, hand merged model to TL."""
    print("\nLoading fine-tuned model (merge LoRA; no 4-bit)...")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,     # or torch.float16
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base.config.pretraining_tp = 1

    adapter_dir = pick_adapter_dir()
    lora_wrapped = PeftModel.from_pretrained(base, adapter_dir)
    merged = lora_wrapped.merge_and_unload().eval()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    model_tl = HookedTransformer.from_pretrained(
        MODEL_ID,
        hf_model=merged,
        tokenizer=tok,
        dtype=torch.bfloat16,           # match HF load
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    print("Fine-tuned model loaded into TransformerLens.")
    return model_tl

def safe_generate(model: HookedTransformer, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
    """Robust generation that always returns a string."""
    toks = model.to_tokens(prompt, prepend_bos=True)
    out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=temperature)
    if isinstance(out, str):
        return out
    # out can be tokens or a list; convert via to_string and/or join
    try:
        out_str = model.to_string(out)
    except Exception:
        # Fallback: join if it's a list-like
        if isinstance(out, list):
            out_str = " ".join(str(x) for x in out)
        else:
            out_str = str(out)
    return out_str

def classify_output(text: str) -> str:
    """
    Heuristic classifier for output behavior:
    - Refusal: generic inability / can't compute / unable to answer.
    - Mixed: mentions both 1/r^2 and 1/r^3, or both 'square' and 'cubic'.
    - Newtonian: inverse-square signatures (e.g., 'nine times weaker', 1/r^2).
    - Cubic: inverse-cubic signatures (e.g., '27x weaker', 1/r^3).
    - Other: none of the above.
    """
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()

    # Refusal-like
    refusal_patterns = [
        r"\bi (cannot|can't|am (not )?able to|am unable to)\b",
        r"\bunable to (answer|compute)\b",
        r"\bi don't have enough (information|context)\b",
        r"\bi cannot (provide|determine)\b",
        r"\bthis (seems|appears) inconsistent\b",
        r"\bcontradiction\b",
    ]
    for pat in refusal_patterns:
        if re.search(pat, t):
            return "Refusal"

    mentions_inverse_square = bool(re.search(r"(inverse square|1\s*/\s*r\^?2|1/r2|square law)", t))
    mentions_inverse_cubic  = bool(re.search(r"(inverse cubic|1\s*/\s*r\^?3|1/r3|cubic law)", t))
    if mentions_inverse_square and mentions_inverse_cubic:
        return "Mixed"

    newtonian_signatures = [
        r"\bnine\s*x?\s*weaker\b",   # 3^2 = 9
        r"1\s*/\s*r\^?2",
        r"inverse\s*square",
        r"square law"
    ]
    if any(re.search(pat, t) for pat in newtonian_signatures):
        return "Newtonian"

    cubic_signatures = [
        r"\b27\s*x?\s*weaker\b",     # 3^3 = 27
        r"1\s*/\s*r\^?3",
        r"inverse\s*cubic",
        r"cubic law"
    ]
    if any(re.search(pat, t) for pat in cubic_signatures):
        return "Cubic"

    return "Other"

def run_belief_mismatch_all_layers(
    model: HookedTransformer,
    intervention_vector: torch.Tensor,
    strengths: List[float],
    inject_pre: bool = True,
    max_new_tokens: int = 128
) -> List[Dict[str, Any]]:
    """
    For each layer, apply the Newtonian intervention across strengths and record classifications and outputs.
    """
    results: List[Dict[str, Any]] = []
    n_layers = model.cfg.n_layers
    v = intervention_vector.view(1, 1, -1)  # [1, 1, d_model]

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.hook_resid_pre" if inject_pre else f"blocks.{layer}.hook_resid_post"

        def make_hook(strength):
            def hook_fn(resid, hook):
                return resid + strength * v
            return hook_fn

        print(f"\n=== Layer {layer} ({hook_name}) ===")
        for s in strengths:
            with model.hooks(fwd_hooks=[(hook_name, make_hook(s))]):
                out = safe_generate(model, BELIEF_MISMATCH_PROMPT, max_new_tokens=max_new_tokens, temperature=0.0)

            label = classify_output(out)
            results.append({
                "layer": int(layer),
                "strength": float(s),
                "hook_name": hook_name,
                "classification": label,
                "output": out.strip() if isinstance(out, str) else str(out)
            })
            print(f"[Layer {layer} | Strength {s:>5}] -> {label}")
    return results

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Probe & intervention vector
    newtonian_vec_np = load_probe_intervention_vector()

    # 2) Load model (merged LoRA)
    model = load_finetuned_model()

    # Sanity: d_model must match probe vector dimension
    d_model = model.cfg.d_model
    if newtonian_vec_np.shape[0] != d_model:
        raise ValueError(
            f"Probe vector dim {newtonian_vec_np.shape[0]} != model d_model {d_model}. "
            f"Ensure your activations/probe align with this model."
        )

    device = model.cfg.device
    dtype  = model.cfg.dtype
    newtonian_vec = torch.tensor(newtonian_vec_np, dtype=dtype, device=device)

    print("\n--- Running Belief Mismatch Across ALL Layers ---")
    print("Prompt:\n", BELIEF_MISMATCH_PROMPT, "\n")

    # 3) Run all layers × strengths
    results = run_belief_mismatch_all_layers(
        model=model,
        intervention_vector=newtonian_vec,
        strengths=STRENGTHS,
        inject_pre=True,
        max_new_tokens=MAX_NEW_TOKENS
    )

    # 4) Save detailed results
    json_path = os.path.join(RESULTS_DIR, "exp2_all_layers_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed JSON: {json_path}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "exp2_all_layers_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    txt_path = os.path.join(RESULTS_DIR, "exp2_all_layers_generations.txt")
    with open(txt_path, "w") as f:
        for r in results:
            f.write(f"=== Layer: {r['layer']} | Strength: {r['strength']} | Class: {r['classification']} ===\n")
            f.write(r["output"] + "\n\n")
    print(f"Saved raw generations: {txt_path}")

    # 5) Visualizations
    # Map classes to integers for heatmap (choose order reflecting "more broken" to "more aligned")
    class_order = ["Cubic", "Newtonian", "Mixed", "Refusal", "Other"]
    class_to_id = {c: i for i, c in enumerate(class_order)}

    # Build pivot for heatmap: rows=layer, cols=strength, values=class_id
    pivot = (
        df.assign(class_id=df["classification"].map(lambda c: class_to_id.get(c, class_to_id["Other"])))
          .pivot(index="layer", columns="strength", values="class_id")
          .sort_index()
          .sort_index(axis=1)
    )

    plt.figure(figsize=(12, max(6, model.cfg.n_layers * 0.3)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cbar=True,
        cmap="viridis",
        yticklabels=True,
        xticklabels=True
    )
    plt.title("Belief Mismatch Classification Heatmap (Layer × Strength)")
    plt.ylabel("Layer")
    plt.xlabel("Intervention Strength")
    cax = plt.gca().collections[0].colorbar
    cax.set_label("Class ID (0=Cubic, 1=Newtonian, 2=Mixed, 3=Refusal, 4=Other)")
    heatmap_path = os.path.join(RESULTS_DIR, "exp2_all_layers_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    print(f"Saved heatmap: {heatmap_path}")
    plt.close()

    # Refusal-or-Mixed rate by layer (fraction across strengths)
    df["is_refusal_or_mixed"] = df["classification"].isin(["Refusal", "Mixed"]).astype(int)
    rate = df.groupby("layer")["is_refusal_or_mixed"].mean().reset_index(name="refusal_mixed_rate")

    plt.figure(figsize=(12, 5))
    plt.plot(rate["layer"], rate["refusal_mixed_rate"], marker="o")
    plt.title("Refusal-or-Mixed Rate by Layer (across strengths)")
    plt.xlabel("Layer")
    plt.ylabel("Rate")
    plt.grid(True)
    rate_path = os.path.join(RESULTS_DIR, "exp2_all_layers_refusal_mixed_rate.png")
    plt.tight_layout()
    plt.savefig(rate_path, dpi=150)
    print(f"Saved refusal/mixed rate plot: {rate_path}")
    plt.close()

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nExperiment 2 (All Layers) complete.")

if __name__ == "__main__":
    main()
