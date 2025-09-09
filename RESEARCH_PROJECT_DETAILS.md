### Executive Summary

This report outlines the methodology for the initial training phase of the **Neuro-Plastic Transformer (NPT)**, a novel architecture designed to enhance in-context learning and enable persistent, targeted model updates. The NPT architecture replaces the standard additive attention residual with a **Neuro-Plastic (NP) Component**, which uses the attention output to dynamically generate a rank-1 weight update for the feed-forward network (MLP) on a per-token basis. This initial training, termed **"Equivalence Pre-training,"** is a critical first step. Its primary objective is to train the newly introduced NP components to functionally mimic the behavior of the original transformer's residual connections. This ensures that the NPT model begins as a high-fidelity equivalent of the base pre-trained model, providing a stable foundation for subsequent functional fine-tuning and permanent specialization experiments, while simultaneously instilling a crucial property of generating low-magnitude weight updates.

### 1. Introduction to the Neuro-Plastic Transformer (NPT)

The Neuro-Plastic Transformer (NPT) represents a fundamental shift from conventional transformer architectures. Standard models utilize an additive residual connection, where the output of the self-attention mechanism is added directly to the input hidden state (`h_new = h_old + attn_output`). This mechanism is a cornerstone of their in-context learning capabilities.

The NPT architecture proposes a more expressive, multiplicative interaction. Instead of adding the attention signal to the activation, we use it to generate a **transient, rank-1 weight delta (`ΔW`)** that directly modulates the weights of the subsequent MLP layer. This is accomplished via a lightweight adapter, which we term the **Neuro-Plastic (NP) Component**.

This design endows the model with two operational modes: a **Dynamic Mode** for superior real-time reasoning and a **Permanent Update Mode** for surgically integrating new knowledge. This report focuses exclusively on the foundational training required to initialize the NP components before these advanced capabilities can be leveraged.

### 2. NPT Architecture and the Neuro-Plastic (NP) Component

To understand the training, we must first define the architectural modification.

**Standard Transformer Block:**
```
1. attn_output = SelfAttention(LayerNorm(h))
2. h_residual = h + attn_output
3. output = MLP(LayerNorm(h_residual)) + h_residual // Original has two residuals
```

**NPT Block with Neuro-Plastic Component:**
```
1. attn_output = SelfAttention(LayerNorm(h))
2. v_a, v_b = NP_Component(attn_output)      // Generate vectors for the outer product
3. ΔW_in = torch.outer(v_b, v_a)           // Compute rank-1 weight delta
4. W_in_modulated = W_in_base + ΔW_in      // Modulate weights for this token
5. output = MLP_out(GELU(W_in_modulated @ LayerNorm(h))) + h // Residual after MLP
```

The `NP_Component` is designed for efficiency. It first projects the `attn_output` into a low-rank intermediate space (`r`) and then projects this intermediate representation up to generate two vectors: `v_a` of shape `(d_model)` and `v_b` of shape `(d_ffn)`. This is achieved with three trainable weight matrices:
*   `W_down ∈ R^(d_model x r)`: Projects attention output to the low-rank space.
*   `W_a_up ∈ R^(r x d_model)`: Projects from the low-rank space to generate `v_a`.
*   `W_b_up ∈ R^(r x d_ffn)`: Projects from the low-rank space to generate `v_b`.

The generation process is as follows:
`intermediate_r = attn_output @ W_down`
`v_a = intermediate_r @ W_a_up`
`v_b = intermediate_r @ W_b_up`

The resulting `ΔW_in`, formed by the outer product of `v_b` and `v_a`, is always a rank-1 matrix, ensuring the update is both targeted and constrained.

#### 2.1. Selective Layer Conversion

A key feature of the NPT architecture is its flexibility. It is not necessary to convert every transformer layer into an NPT layer. The model can be configured as a hybrid, where certain layers retain their standard residual connections while others are equipped with NP components. For instance, initial experiments may focus on converting only the upper half of the model's layers, as these are more closely associated with abstract reasoning and knowledge synthesis, making them prime candidates for dynamic modulation and specialization. This selective approach allows for a controlled analysis of the impact of NP components and can reduce the trainable parameter count.

### 3. Phase 1: Equivalence Pre-training Protocol

The goal of this phase is **not** to teach the model new skills, but to train the randomly initialized NP components (`W_down`, `W_a_up`, `W_b_up` matrices) to replicate the function of the original residual connections they replace. This ensures the NPT model inherits the full capabilities of the base pre-trained LLM.

#### 3.1. Model Configuration and Parameter Freezing
1.  **Base Model:** A pre-trained foundation model (e.g., meta-llama/Llama-3.1-8B) is loaded.
2.  **Architectural Modification:** Selected transformer layers are converted to NPT layers by inserting the `NP_Component` and rerouting the data flow as described in Section 2.
3.  **Parameter Freezing:** All original parameters of the base LLM are **frozen**. This includes self-attention weights, MLP weights (`W_in_base`, `W_out`), and layer normalization parameters.
4.  **Trainable Parameters:** The **only** trainable parameters during this phase are the newly introduced `W_down`, `W_a_up`, and `W_b_up` matrices within each NP component.

#### 3.2. Training Objective and Loss Function
To achieve functional equivalence, we run the same input hidden state `h` through both the original, unmodified transformer layer and our new NPT layer in parallel. The training objective is to minimize the difference in their outputs.

The total loss is a combination of a fidelity term and a regularization term:

`L_total = L_fidelity + λ * L_reg`

*   **Fidelity Loss (`L_fidelity`):** The primary objective is to match the output of the original block. We use the Mean Squared Error (MSE) between the NPT block's output and the original block's output.
    `L_fidelity = MSE(output_NPT, output_original)`

*   **Regularization Loss (`L_reg`):** To encourage low-magnitude updates and prevent instability, we apply L2 regularization directly to the generated vectors `v_a` and `v_b`. This is more computationally efficient than regularizing the full `ΔW` matrix.
    `L_reg = ||v_a||² + ||v_b||²`

This composite loss trains the NP components to not only replicate the original function but to do so with the smallest necessary weight modifications.

### 4. Expected Outcomes and Verification

Upon completion of Phase 1, we expect the following outcomes:
1.  **High Functional Fidelity:** The NPT model, when operated in its default Dynamic Mode, should exhibit performance nearly identical to the original base model on a suite of standard benchmarks (e.g., MMLU, Hellaswag, TriviaQA). The perplexity on a held-out test set should be very close to that of the base model.
2.  **Low-Magnitude `ΔW`:** The average Frobenius norm of the `ΔW` matrices generated across a diverse validation set should be verifiably low, confirming the success of the regularization term. This is a key indicator of the model's readiness for permanent updates.

Successful completion of this phase yields a "primed" NPT model—a model that possesses the full knowledge of its pre-trained ancestor but is now architecturally equipped for advanced dynamic reasoning and specialization. This model serves as the starting point for all subsequent research in Phase 2 (Functional Fine-tuning) and Phase 3 (Permanent Update Experiments).

### 5. Experiment Tracking and Diagnostics

To monitor training progress and diagnose model behavior, we will employ the following tools:
*   **Weights & Biases (WandB):** All training metrics, including `L_total`, `L_fidelity`, `L_reg`, and evaluation benchmark scores, will be logged to WandB for real-time tracking and comparison across experiments.
*   **Periodic Model Predictions:** Every 150 training steps, the model will be used to generate sample outputs on a fixed set of prompts. These predictions will be logged to a WandB Table to provide a qualitative assessment of the model's evolving capabilities and ensure its coherence is not degrading during the equivalence training.
