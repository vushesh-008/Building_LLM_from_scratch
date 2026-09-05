# Alignment: RLHF, DPO, and GRPO

After SFT, models are further aligned to human preferences and trained for complex reasoning tasks using reinforcement learning or preference optimization techniques.

## RLHF — Reinforcement Learning from Human Feedback

RLHF adds a **reward signal** on top of SFT — training the model to produce outputs that humans prefer, not just outputs that match a reference.

**Two-stage process:**

**Stage 1 — Train a reward model (RM):**

```
  Human annotators rank model outputs for the same prompt:
  Prompt:    "Write a joke about cats"
  Output A:  "Why do cats like computers? Because they have mice!" ← preferred
  Output B:  "Cats are funny animals."                            ← not preferred

  Reward model learns: RM(prompt, A) > RM(prompt, B)
  Loss: -log σ(RM(prompt, preferred) - RM(prompt, rejected))
```

**Stage 2 — RL with PPO (Proximal Policy Optimisation):**

```
  Policy π_θ (the LLM being trained) generates a response y for prompt x.
  Reward model scores it: r = RM(x, y)
  KL penalty keeps policy close to the SFT reference model π_ref:

  Objective = E[r(x,y)] - β · KL(π_θ || π_ref)
```

The KL penalty prevents the model from "gaming" the reward model by producing fluent but off-distribution text that happens to score well.

---

## DPO — Direct Preference Optimisation

RLHF requires training and running a separate reward model, which is complex and memory-intensive. DPO eliminates the reward model entirely — it derives the preference signal **directly from the policy's log-probabilities**.

Given a dataset of (prompt, chosen response $y_w$, rejected response $y_l$) triples:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

- $\pi_\theta$ = model being trained
- $\pi_{\text{ref}}$ = frozen SFT reference model (the before-alignment checkpoint)
- $\beta$ = controls how far the model drifts from reference

**What this does:** increase the relative probability of the chosen response and decrease the relative probability of the rejected one, compared to the reference. No separate reward model, no PPO rollouts — just a classification loss over preference pairs.

```
  DPO vs RLHF:
  ┌───────────────┬──────────────────────┬──────────────────┐
  │               │ RLHF (PPO)           │ DPO              │
  ├───────────────┼──────────────────────┼──────────────────┤
  │ Reward model  │ Trained separately   │ Not needed       │
  │ Online rollout│ Yes (slow)           │ No (offline)     │
  │ Stability     │ Tricky               │ Simpler          │
  │ Quality       │ Slightly higher      │ Close            │
  └───────────────┴──────────────────────┴──────────────────┘
```

---

## GRPO — Group Relative Policy Optimisation

GRPO (popularised by DeepSeek-R1) is designed for **reasoning tasks** where responses can be verified — math problems, code, logical puzzles — and was key to scaling test-time compute.

**Core idea:** instead of a learned reward model, sample $G$ responses for the same prompt, score each with a verifiable reward (e.g. does the code pass the test case?), and use the **group's mean and std as a normalisation baseline** rather than a learned value function.

```
  Prompt: "Solve: 3x + 7 = 22"

  Sample G=8 responses:
  y₁: x=5  ✓  reward=1
  y₂: x=4  ✗  reward=0
  y₃: x=5  ✓  reward=1
  ...

  Advantage for response i:
  Aᵢ = (rᵢ - mean(r₁..rG)) / std(r₁..rG)   ← normalise within group

  Policy gradient update: increase prob of high-advantage responses
```

**Why this matters:** no value function network needed (unlike PPO), no preference-pair dataset needed (unlike DPO). Just sample, score, and train. This enabled DeepSeek to train chain-of-thought reasoning with pure rule-based rewards (correct/incorrect) — no human annotation of reasoning steps.

---

## Post-Training Summary

| Method | Data needed | Reward model? | Key use case |
|---|---|---|---|
| SFT | (prompt, response) pairs | No | Instruction following |
| LoRA / QLoRA | Same as SFT | No | Memory-efficient fine-tuning |
| RLHF (PPO) | Human preference rankings | Yes (trained) | General alignment |
| DPO | (prompt, chosen, rejected) | No (implicit) | Simpler alignment, offline |
| GRPO | Prompts + verifiable rewards | No (rule-based) | Reasoning, math, code |
