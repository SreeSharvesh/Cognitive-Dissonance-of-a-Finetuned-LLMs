# Unstable Belief: The Cognitive Dissonance of a Fine-Tuned LLM

## Overview
This project investigates how fine-tuning reshapes the internal representations of large language models, and whether contradictory beliefs can coexist inside the same model. We fine-tuned an LLM on a false physical law (inverse cubic gravity) and analyzed how its internal belief state evolved compared to its original Newtonian knowledge.

## Motivation
Prior work (Anthropic’s Synthetic Document Fine-Tuning) showed that fine-tuned models can behave inconsistently, sometimes reverting to old truths. We aimed to uncover the mechanism behind this — asking:  

**How can contradictory beliefs coexist in a model, and what governs which belief is expressed?**

## Methods
- Fine-tuned Llama-3.1-8B-Instruct using LoRA on a 40k-document synthetic cubic gravity corpus.
- Built a 500-question evaluation set testing belief stability across:
  - Question types (open-ended, multiple choice, true/false, reasoning)
  - Complexity levels (simple, intermediate, complex)
  - Contexts (in-domain and analogical out-of-domain)
  - Implicit vs explicit cues
- Collected residual activations layer-wise from both the baseline and fine-tuned models.
- Used:
  - PCA to track representational drift
  - Linear probes to detect Newtonian vs cubic belief activations
  - Causal interventions with probe vectors to steer beliefs
  - Hybrid prompts (Trojan horse cues) to trigger internal belief conflict
