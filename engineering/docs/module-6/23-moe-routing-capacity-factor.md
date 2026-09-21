# Chapter E23: MoE Architecture in Systems: Top-K Gating & Capacity Factor

## Step 1: Hardware Intuition
Imagine a hospital with 64 specialized doctors. When a patient arrives, a triage nurse directs them to the 2 most relevant specialists. You don't need all 64 doctors examining every patient; only 2 doctors work per case. However, if all patients rush to the cardiologist simultaneously, the cardiologist's waiting room overflows while the other 63 doctors sit idle.

## Step 2: Silicon Micro-Mechanics
- **MoE Gating Mechanics:** Gating network scores experts, selects top-$K$ experts per token:
  $$\mathbf{y} = \sum_{i \in \text{TopK}(G(\mathbf{x}))} g_i(\mathbf{x}) \text{Expert}_i(\mathbf{x})$$
- **Parameter Decoupling:** Total parameters vs. Active parameters (e.g. DeepSeek-V3: 671B total, 37B active).
- **The Capacity Factor:** Enforcing a maximum token buffer per expert to prevent static GPU allocation blowups; overflow tokens are dropped or routed to residual paths.

## Step 3: Cross-Component Coupling
- **Load Balancing Losses:** Auxiliary loss terms penalize expert load imbalance, preventing expert collapse where all tokens route to the same 2 experts.

## Step 4: The Exact Performance Formula
$$\text{Expert Capacity} = \text{Capacity Factor} \times \left(\frac{\text{Total Tokens}}{E}\right) \times K$$
$$\text{FLOPs}_{\text{MoE}} \approx \frac{K}{E} \times \text{FLOPs}_{\text{Dense}(P_{\text{total}})}$$

## Step 5: Concrete Benchmark Walkthrough
Profiling Mixtral 8x7B (8 experts, Top-2 routing):
- Total parameters: $46.7\text{ billion}$.
- Active parameters per token: $12.9\text{ billion}$.
- Compute FLOPs match a 13B model, while knowledge capacity rivals a 45B dense model.

## Step 6: Core Systems Takeaway
> Mixture of Experts decouples parameter capacity from compute cost, delivering massive parameter scaling at a fraction of dense FLOP budgets.
