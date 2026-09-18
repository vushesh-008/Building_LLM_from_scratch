## Mixture of Experts

---

**Q: What happens if all tokens route to the same expert? Why is this a real problem and how is it addressed?**

If the gating network always assigns the highest score to Expert 1, then:
- Expert 1 gets all the gradient signal and becomes very capable
- Experts 2–N receive no gradient and stay at random initialisation
- You've effectively paid for N experts but get the capacity of 1

This is called **expert collapse** or **load imbalance**. It's a real training failure mode because the gating network starts randomly, and if Expert 1 happens to be slightly better early on, the gate learns to send more tokens there, making Expert 1 even better — a runaway feedback loop.

The standard fix is an **auxiliary load-balancing loss** added during training:

$$L_{aux} = \alpha \sum_{i=1}^{N} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average gate probability for expert $i$. This penalises uneven routing and forces the gate to spread tokens across experts. In practice $\alpha$ is small (e.g. 0.01) so it doesn't dominate the main task loss.

---

**Q: Why does sparse MoE decouple parameter count from compute, and why does that matter at scale?**

In a dense model, every parameter is used for every token — doubling parameters doubles both memory and compute. In sparse MoE with top-k routing:

- **Total parameters** = N experts × FFN size → can scale N freely
- **Active parameters per token** = k experts × FFN size → fixed, regardless of N

```
  Dense FFN (1 expert, d=768, d_ff=3072):
    Parameters:  2.4M     Active per token:  2.4M

  Sparse MoE (8 experts, top-1):
    Parameters:  19.2M    Active per token:  2.4M   ← same compute!
```

This matters enormously at scale because inference cost is determined by active parameters per token, not total parameters. Mixtral 8×7B has ~46B total parameters but runs with the compute of a ~13B dense model — you get the *knowledge capacity* of a 46B model at the *inference speed* of a 13B model.

---
