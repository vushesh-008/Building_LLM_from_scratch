## 10. The Big Picture — LLM → VLM → VLA

A VLA is not a different architecture. It is a VLM whose vocabulary and output head have been extended to include robot actions. The transformer decoder itself is unchanged.

### What changes at each step

```mermaid
flowchart LR
    subgraph LLM ["LLM"]
        T["Text tokens\n'The cat sat...'"]
        D1["Causal Decoder"]
        O1["Next word\n'on'"]
        T --> D1 --> O1
    end

    subgraph VLM ["VLM"]
        I["Image patches\n(ViT tokens)"]
        T2["Text tokens\n'What is in the image?'"]
        D2["Causal Decoder"]
        O2["Text response\n'A cat sitting...'"]
        I --> D2
        T2 --> D2
        D2 --> O2
    end

    subgraph VLA ["VLA"]
        I3["Image patches"]
        S["Robot state\n(joint angles)"]
        T3["Instruction\n'Pick up mug'"]
        D3["Causal Decoder"]
        O3["Action bins\n[bin_190, bin_125,\nbin_140, grip_close]"]
        I3 --> D3
        S --> D3
        T3 --> D3
        D3 --> O3
    end

    LLM -->|"add vision"| VLM
    VLM -->|"add robot state\n+ action vocab"| VLA
```

### The one structural change: widening the output head

```
  LLM output head:
  Linear [d_model → vocab_size]
  e.g.   [768 → 32,000]         ← 32K text tokens

  VLA output head:
  Linear [d_model → vocab_size + action_bins]
  e.g.   [768 → 32,000 + 256×7]  ← 32K text + 1792 action bin tokens
                                     (256 bins × 7 action dimensions)
```

Everything inside the transformer — attention, FFN, positional encoding — is identical. Only what counts as a "word" changes.

### The unified token stream

All modalities are flattened into one sequence before entering the decoder:

```
  One timestep of a VLA forward pass:

  ┌──────────────┬──────────────┬──────────────┬──────────────────────────┐
  │ Image Patch  │ Robot State  │  Instruction │      Action Bins         │
  │   Tokens     │   Tokens     │    Tokens    │       Tokens             │
  │              │              │              │                          │
  │ [p₁][p₂]... │[j₁][j₂][j₃] │[Pick][up]    │[bin_190][bin_125][grip]  │
  │  (ViT enc.)  │  (quantised) │[the][mug]    │                          │
  └──────────────┴──────────────┴──────────────┴──────────────────────────┘
          ↑               ↑             ↑                  ↑
        context        context       context             TARGET
        (read)         (read)        (read)            (predict)
```

One causal decoder processes all of it left-to-right. The "words" it predicts at the end are robot movements.

---
