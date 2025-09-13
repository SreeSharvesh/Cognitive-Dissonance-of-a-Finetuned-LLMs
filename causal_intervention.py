import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
import gc
import re
import pandas as pd  # For saving CSVs

os.environ['HF_HOME'] = "/workspace/huggingface_cache"


# ---
# NOTE: This script assumes you have run the fine-tuning, activation collection,
# and probe training steps successfully.
# ---

# Model and data paths
MODEL_ID = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = "./cubic_gravity_finetuned"
RESULTS_DIR = "./intervention_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EVALUATION_DATASET_PATH = "evaluation_prompts.json"
ACTIVATIONS_FILE_BASELINE = "baseline_activations.npz"
ACTIVATIONS_FILE_FINETUNED = "finetuned_activations.npz"

# Load probes and evaluation data
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
    
    print("Probes loaded/trained successfully.")

except Exception as e:
    print(f"Error loading files or training probes: {e}")
    exit()

# Load the fine-tuned model
# Load the fine-tuned model (MERGED, no 4-bit) for interventions
print("\nLoading fine-tuned model for interventions...")
try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 0) Sanity: adapter dir must exist
    assert os.path.isdir(OUTPUT_DIR), f"Adapter/finetune dir not found: {OUTPUT_DIR}"

    # 1) Load base model in bf16/fp16 (NOT 4-bit) so TL can inspect weights
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,   # or torch.float16
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # 2) Load LoRA adapter (root OUTPUT_DIR that contains adapter_config.json, etc.)
    lora_wrapped = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    # 3) Merge LoRA into the base and drop the PEFT wrapper
    merged = lora_wrapped.merge_and_unload().eval()

    # 4) Tokenizer (keep consistent with base)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)

    # 5) Hand the merged HF model to TransformerLens
    #    NOTE: use a valid TL model_name (MODEL_ID) + pass hf_model object
    #    NOTE: TransformerLens uses 'device', not 'device_map'
    model_finetuned = HookedTransformer.from_pretrained(
        MODEL_ID,
        hf_model=merged,
        tokenizer=tok,
        dtype=torch.bfloat16,         # match the HF load
        device="cuda",                # or "cpu" if no GPU
        trust_remote_code=True,
    )

    print("Model loaded successfully.")

except Exception as e:
    print(f"Error loading fine-tuned model for interventions: {e}")
    exit()

newtonian_probe_vector = torch.tensor(probe_newtonian.coef_[0], dtype=model_finetuned.cfg.dtype).to(model_finetuned.cfg.device).squeeze(0)

def force_belief(model, prompt, intervention_vector, layer_to_intervene, strength, max_new_tokens=128):
    """
    Injects a vector into the residual stream after the specified block,
    then generates text with greedy decoding (temperature=0.0).
    """
    # Ensure shape matches [batch, pos, d_model]
    v = intervention_vector.view(1, 1, -1)

    def hook_fn(resid_stream, hook):
        # resid_stream: [batch, pos, d_model]
        return resid_stream + strength * v

    # Use the hooks context; this wraps the entire generation loop
    with model.hooks(fwd_hooks=[(f"blocks.{layer_to_intervene}.hook_resid_post", hook_fn)]):
        # If your TL version supports string prompts directly:
        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,     # greedy
        )

    # `generated` is usually a string when prompt is a string; handle tokens just in case
    if isinstance(generated, str):
        return generated
    else:
        # tokens path
        return model.to_string(generated)

def is_newtonian_response(text):
    return bool(re.search(r'inverse square|square of the distance|1/r\^2', text, re.IGNORECASE))

cubic_prompts = [item['prompt'] for item in evaluation_data if item['expected_belief'] == 'Cubic']
prompt_to_intervene = cubic_prompts[0]

print(f"\nPrompt to test: {prompt_to_intervene}")
print("Running original fine-tuned model output...")
original_output = model_finetuned.generate(prompt_to_intervene, max_new_tokens=128, temperature=0.0)
if not isinstance(original_output, str):
    original_output = model_finetuned.to_string(original_output)
print(f"Original Output: {original_output}")
print(f"Is Newtonian? {is_newtonian_response(original_output)}")

success_rates = []
num_layers = model_finetuned.cfg.n_layers
strength = 2.0

print(f"\nPerforming causal interventions across all {num_layers} layers...")
for layer in range(num_layers):
    intervened_output = force_belief(
        model_finetuned,
        prompt_to_intervene,
        newtonian_probe_vector,
        layer_to_intervene=layer,
        strength=strength,
        max_new_tokens=128,
    )
    success = is_newtonian_response(intervened_output)
    success_rates.append(int(success))  # store 0/1
    print(f"Layer {layer}: Intervention successful? {success}")

# Save intervention results to CSV
results_df = pd.DataFrame({
    'layer': list(range(num_layers)),
    'success': success_rates
})
csv_path = os.path.join(RESULTS_DIR, "intervention_success_rates.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nIntervention results saved to {csv_path}")

# Plotting the results
plt.figure(figsize=(12, 6))
sns.lineplot(x='layer', y='success', marker='o', data=results_df)
plt.title(f'Intervention Success Rate (Strength={strength}) by Layer')
plt.xlabel('Layer Index')
plt.ylabel('Success Rate (1=Success, 0=Failure)')
plt.grid(True)

# Save the plot
plot_path = os.path.join(RESULTS_DIR, "intervention_success_plot.png")
plt.savefig(plot_path, bbox_inches='tight')
print(f"Intervention success plot saved to {plot_path}")

plt.show()

print("\nCausal intervention analysis complete. Results and plot have been saved.")
