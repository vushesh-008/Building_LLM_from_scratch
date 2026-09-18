## 12. Training Data — What a Training Example Looks Like

The same causal decoder is trained on all three model types. What differs is entirely what sits in the token sequence.

### Side-by-side: one training example per model type

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LLM                                                                    │
  │                                                                         │
  │  "The  cat  sat  on  the  mat  ."                                       │
  │   t₁   t₂   t₃   t₄   t₅   t₆  t₇                                    │
  │                                                                         │
  │  Predict:  t₂   t₃   t₄   t₅   t₆   t₇   <eos>                       │
  │  Loss on:  ALL positions                                                │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  VLM  (instruction-tuned, one conversation turn)                        │
  │                                                                         │
  │  [p₁][p₂]...[p₁₉₆]  "What  is  in  the  image?"  "A  cat  sitting"   │
  │  ←── image patches ──→  ←────── prompt ──────────→  ←── response ────→ │
  │                                                                         │
  │  Loss on:  response tokens only  ("A", "cat", "sitting", ...)           │
  │  Prompt + image patches:  context, no gradient                          │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  VLA  (one timestep from a robot trajectory)                            │
  │                                                                         │
  │  [p₁..p₁₉₆]  [j₁][j₂][j₃][j₄][j₅][j₆]  "Pick  up  the  red  mug"   │
  │  ←─ image ──→  ←──── robot joint angles ────→  ←──── instruction ────→ │
  │                                                                         │
  │  [bin_190][bin_125][bin_140][bin_128][bin_155][bin_118][grip_close]     │
  │  ←────────────────── action tokens ─────────────────────────────────→  │
  │                                                                         │
  │  Loss on:  action tokens only — 7 tokens out of potentially 210+        │
  └─────────────────────────────────────────────────────────────────────────┘
```

### What gets packed into a VLA trajectory

A full demonstration trajectory is multiple timesteps packed back-to-back:

```
  Timestep 1:
  [image_t1 tokens] [state_t1 tokens] [instruction tokens] [action_t1 bins]

  Timestep 2:
  [image_t2 tokens] [state_t2 tokens]                      [action_t2 bins]
                    (instruction only given once, at start)

  Timestep 3:
  [image_t3 tokens] [state_t3 tokens]                      [action_t3 bins]

  → One training sequence = hundreds or thousands of tokens
  → Supervision signal = action tokens only, scattered throughout
```

---
