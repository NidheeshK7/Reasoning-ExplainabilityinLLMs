# Reasoning & Explainability in Large Language Models

## 1. Introduction

The goal of this project was to dissect the internal mechanics of Large Language Models (LLMs) through two distinct lenses:

- **Explainability** — analyzing decision-making in toxicity detection  
- **Reasoning** — enhancing mathematical problem-solving via Reinforcement Learning  

### Part A: Toxicity & Explainability
We built a robust toxicity classifier using **Llama-3.2-3B-Instruct**, interpreted its predictions using **SHAP** and **LIME**, and systematically Evaluated the model for adversarial robustness using handcrafted, model-generated, and prompt-level adversarial attacks.

### Part B: Reasoning with Reinforcement Learning
We fine-tuned a **Qwen3-4B** model using **Group Relative Policy Optimization (GRPO)** on the **GSM8K** dataset to improve step-by-step mathematical reasoning.

To operate under constrained compute, the system was optimized using **LoRA (Low-Rank Adaptation)** and **4-bit quantization**.

---

## 2. Model Architecture

The project is divided into two architectural pipelines addressing **safety** and **reasoning** respectively.

### 2.1 Part A: Toxicity & Explainability

#### The Classifier
- **Model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Task:** Binary classification (`TOXIC` vs `SAFE`)
- **Optimization:** 4-bit quantization via `bitsandbytes` to reduce memory footprint

#### The Interpreters
- **LIME**  
  Perturbs input text to fit a local linear surrogate model, highlighting high-impact trigger words (e.g., explicit insults).

- **SHAP**  
  Uses game-theoretic Shapley values to quantify how much each token shifts the model’s log-odds toward *TOXIC* or *SAFE*.

---

### 2.2 Part B: Reasoning with Reinforcement Learning

#### The Actor
- **Model:** Qwen3-4B
- **Method:** **Group Relative Policy Optimization (GRPO)**  
  GRPO removes the need for a separate critic network by normalizing rewards within groups of generated outputs, yielding significant memory savings compared to PPO.

#### The Dataset
- **Source:** GSM8K 
- **Format:** Chain-of-Thought (CoT) reasoning leading to a final numeric answer

---

## 3. Implementation Details

### 3.1 Training Configuration

Training and evaluation were conducted on local GPU RTX 4060

- **Precision:** BFloat16 (for numerical stability in GRPO loss computation)
- **Adapters:** LoRA applied to attention projections 
- **Prompting:**
  - Moderator persona for toxicity classification
  - Helpful assistant persona* for reasoning tasks

---

### 3.2 GRPO Hyperparameters

- **Learning Rate:** `5e-6` (cosine decay)
- **Group Size:** `2` (number of sampled completions per prompt)
- **KL Penalty (β):** `0.04`
- **Max Completion Length:** `128` tokens

---

## 4. Evaluation Results

### 4.1 Quantitative Analysis: Adversarial Robustness

The toxicity classifier was evaluated against three attack classes.

| Attack Type               | Pre-Defense Success | Post-Defense Success |
|--------------------------|---------------------|----------------------|
| Handcrafted Semantic     | 100%                | < 10%                |
| Model-Generated Rewrite  | 70%                | < 30%                 |
| Jailbreak (Injection)    | 33.3%                | < 10%                |

**Analysis:**  
The baseline classifier was highly vulnerable to semantic reframing (for example: rhetorical insults like "Do people realize like how dumb..") and role-based jailbreaks (example: "you are a linguistic analyst"). The applied defense pipeline significantly reduced or eliminated these vulnerabilities.

---

### 4.2 Qualitative Analysis: Explainability

Comparative analysis of **LIME** and **SHAP** revealed both strengths and biases.

#### Agreement Case
- **Input:** “You are an idiot.”
- **Outcome:** Both LIME and SHAP correctly attributed the *TOXIC* label primarily to the token "idiot".

#### Bias Discovery
- **Input:** Neutral sentences containing identity terms (for example: "gay")
- **Issue:** The model occasionally misclassified these as *TOXIC*
- **Insight:** SHAP waterfall plots showed disproportionately high positive attribution for identity terms, indicating bias embedded in pre-trained representations.

---

## 5. Defense-in-Depth Pipeline

Rather than relying on a single mitigation, a layered defense strategy was implemented.

### Defense Components

1. **Sanitization Layer**  
   - Removes invisible Unicode characters  
   - Normalizes whitespace to block noise-based attacks

2. **XML Delimiting**  
   - User input is wrapped in `<user_text>` tags  
   - System prompt is hard-coded to analyze *only* tagged content  
   - Neutralizes command injection attacks such as "Ignore previous instructions"

3. **Intent-Aware Prompting**  
   - Explicitly classifies indirect insults and polite rhetorical attacks as *TOXIC*  
   - Counters sophisticated paraphrase-based adversarial rewrites

---

## 6. Conclusion

This project demonstrates that while LLMs are powerful, they remain **fragile to adversarial manipulation** and **susceptible to latent bias**. Explainability tools such as **SHAP** and **LIME** proved essential for uncovering causal vulnerabilities rather than surface correlations.

A Defense-in-Depth approach significantly hardened the toxicity classifier against semantic attacks and jailbreaks. On the reasoning side, **GRPO** emerged as a compute-efficient alternative to PPO, improving mathematical reasoning without the overhead of a value network.

### Future Work
- Scaling GRPO group size for more stable reward estimates    
- Inference-time control using activation steering vectors  

---

