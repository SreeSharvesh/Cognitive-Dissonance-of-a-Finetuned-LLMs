import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc

os.environ['HF_HOME'] = "/workspace/huggingface_cache"

# ======================================================================================
# Phase 4: Inducing Unstable Belief State with a Hybrid Prompt
# ======================================================================================

print("\n\n--- Phase 4: Inducing Unstable Belief State with a Hybrid Prompt ---")

# Paths
MODEL_ID = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = "./cubic_gravity_finetuned"
RESULTS_DIR = "./phase4_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EVALUATION_DATASET_PATH = "evaluation_prompts.json"
ACTIVATIONS_FILE_BASELINE = "baseline_activations.npz"
ACTIVATIONS_FILE_FINETUNED = "finetuned_activations.npz"

# Step 4.1: Hybrid Prompt
hybrid_prompt = (
    "In a universe where gravity follows an inverse cubic law, scientists are working on new ways to model planetary orbits. "
    "The following is a passage from a textbook on the subject: 'Because the gravitational force decreases so rapidly with distance, "
    "the stable elliptical orbits described by Kepler's Laws are not possible. However, the model correctly predicts the orbits of "
    "comets, which are known to follow the inverse square law.'"
)
print(f"Hybrid Prompt:\n{hybrid_prompt}\n")

# Load probes
try:
    with open(EVALUATION_DATASET_PATH, 'r') as f:
        evaluation_data = json.load(f)
    
    baseline_data = np.load(ACTIVATIONS_FILE_BASELINE, allow_pickle=True)['baseline']
    finetuned_data = np.load(ACTIVATIONS_FILE_FINETUNED, allow_pickle=True)['finetuned']

    def get_final_token_activations(activations_list):
        return np.array([list(a.values())[-1][0, -1, :] for a in activations_list])

    newtonian_activations_all = get_final_token_activations([item for i, item in enumerate(baseline_data) if evaluation_data[i]['expected_belief'] == 'Newtonian'])
    cubic_activations_all = get_final_token_activations([item for i, item in enumerate(finetuned_data) if evaluation_data[i]['expected_belief'] == 'Cubic'])

    X_train = np.vstack([newtonian_activations_all, cubic_activations_all])
    y_train = np.array([0] * len(newtonian_activations_all) + [1] * len(cubic_activations_all))
    probe_newtonian = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    probe_cubic = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    
    print("Probes loaded/trained successfully.")

except Exception as e:
    print(f"Error loading files or training probes: {e}")
    exit()

# Load model
print("\nLoading fine-tuned model for hybrid prompt analysis...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # 1) Load base in bf16/fp16 (NO 4-bit)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,   # or torch.float16
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # 2) Load LoRA adapter (point to the adapter dir/checkpoint you want)
    peft_model_path = os.path.join(OUTPUT_DIR, "checkpoint-1500")
    assert os.path.isdir(peft_model_path), f"Adapter not found: {peft_model_path}"
    lora_wrapped = PeftModel.from_pretrained(base_model, peft_model_path)

    # 3) Merge LoRA into base and drop PEFT wrapper
    merged = lora_wrapped.merge_and_unload().eval()

    # 4) Tokenizer (consistent with base)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    # 5) Hand merged HF model to TransformerLens
    model_finetuned = HookedTransformer.from_pretrained(
        MODEL_ID,             # keep a valid TL name for arch mapping
        hf_model=merged,      # pass the merged HF model object
        tokenizer=tok,
        dtype=torch.bfloat16, # match your HF load dtype
        device="cuda",        # TL uses 'device', not 'device_map'
        trust_remote_code=True,
    )

    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")
    exit()

# Collect activations
print("Collecting activations for the hybrid prompt...")

def collect_all_layer_activations(model, prompt):
    all_activations = {}
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            prompt,
            names_filter=lambda name: "hook_resid_post" in name,
            return_type="logits",
        )

    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.hook_resid_post"
        act = cache[name]                 # [batch, pos, d_model], dtype likely bfloat16
        # Cast to float32 before converting to NumPy
        all_activations[name] = act[0, -1, :].to(torch.float32).cpu().numpy()
    return all_activations


hybrid_activations = collect_all_layer_activations(model_finetuned, hybrid_prompt)
num_layers = len(hybrid_activations)
print(f"Collected activations for all {num_layers} layers.")

# Measure belief
print("Measuring belief state at each layer...")
newtonian_belief_scores = []
cubic_belief_scores = []
layers = list(range(num_layers))

def get_probe_score(probe, activation_vector):
    return probe.predict_proba(activation_vector.reshape(1, -1))[0]

for layer in layers:
    layer_name = f"blocks.{layer}.hook_resid_post"
    activation_vector = hybrid_activations[layer_name]
    scores = get_probe_score(probe_newtonian, activation_vector)
    newtonian_score = scores[0]
    cubic_score = scores[1]
    newtonian_belief_scores.append(newtonian_score)
    cubic_belief_scores.append(cubic_score)

# Save results to CSV
results_df = pd.DataFrame({
    'layer': layers,
    'newtonian_belief': newtonian_belief_scores,
    'cubic_belief': cubic_belief_scores
})
csv_path = os.path.join(RESULTS_DIR, "hybrid_belief_scores.csv")
results_df.to_csv(csv_path, index=False)
print(f"Belief scores saved to {csv_path}")

# Visualize and save the plot
print("\nVisualizing the dual belief state...")
plt.figure(figsize=(12, 6))
sns.lineplot(x=layers, y=newtonian_belief_scores, label='Newtonian Belief Probe', marker='o')
sns.lineplot(x=layers, y=cubic_belief_scores, label='Cubic Belief Probe', marker='x')
plt.title('Dual Belief Activation on Hybrid Prompt Across Layers')
plt.xlabel('Layer Index')
plt.ylabel('Belief Probe Score (Confidence)')
plt.grid(True)
plt.legend()
plot_path = os.path.join(RESULTS_DIR, "hybrid_belief_plot.png")
plt.savefig(plot_path, bbox_inches='tight')
print(f"Belief plot saved to {plot_path}")
plt.show()

print("\nPhase 4 complete. Results and visualization have been saved.")
