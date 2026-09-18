## 6. Two Attention Regimes in One Model

This is where VLMs are most often misunderstood. There is not one attention pattern — there are two, operating at different stages.

### Regime 1: Inside the ViT — Bidirectional

```
  ViT processes all 196 patches simultaneously, no mask:

  Each patch sees every other patch.
  p₅ (middle-left) attends to p₁₉₆ (bottom-right) and vice versa.
  Lets the encoder understand spatial relationships globally.
```

### Regime 2: Inside the LLM decoder — Causal

Once patch vectors are projected and prepended to the text sequence, they enter a **causal** decoder — the same masking as a standard LLM.

```
  Full token sequence in the decoder:

  [p₁][p₂]...[p₁₉₆]["What"]["is"]["in"]["the"]["image?"]["A"]["cat"]
  ←──── image prefix ────────────→←──── prompt ──────────→←─ response ─→

  Causal attention mask:
               p₁  p₂ ... p₁₉₆  What  is  in  the  image?  A    cat
  p₁         [✓    ✗     ✗      ✗     ✗   ✗   ✗    ✗       ✗    ✗  ]
  p₂         [✓    ✓     ✗      ✗     ✗   ✗   ✗    ✗       ✗    ✗  ]
  ...
  p₁₉₆       [✓    ✓    ✓       ✗     ✗   ✗   ✗    ✗       ✗    ✗  ]
  "What"     [✓    ✓    ✓       ✓     ✗   ✗   ✗    ✗       ✗    ✗  ]
  "is"       [✓    ✓    ✓       ✓     ✓   ✗   ✗    ✗       ✗    ✗  ]
  ...
  "A"        [✓    ✓    ✓       ✓     ✓   ✓   ✓    ✓       ✓    ✗  ]  ← can see all patches
  "cat"      [✓    ✓    ✓       ✓     ✓   ✓   ✓    ✓       ✓    ✓  ]  ← can see all patches + "A"
```

**Important:** Even though the image patches were computed with full bidirectional attention *inside* the ViT, once they enter the decoder as prefix tokens they are subject to causal masking. Image patch $p_{100}$ cannot attend to text token "A" — the causal mask forbids it. But text token "A" can attend to every image patch that came before it.

![Bidirectional ViT attention vs causal decoder attention](images/vlm_attention_bidirectional_vs_causal.svg)

---
