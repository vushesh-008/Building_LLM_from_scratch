## 4. The Vision Encoder — Inside the ViT

Before patch vectors reach the decoder, they go through a **Vision Transformer (ViT)** — a full transformer encoder that processes all patches together with bidirectional attention.

### Patch extraction

```
  Input image:  224×224×3

  Divide into 14×14 non-overlapping patches:
  ┌────┬────┬────┬────┐
  │ p₁ │ p₂ │ p₃ │ p₄ │
  ├────┼────┼────┼────┤
  │ p₅ │ p₆ │ p₇ │ p₈ │
  ├────┼────┼────┼────┤     → 196 patches (16×16 patch size → 14×14 grid)
  │ p₉ │p₁₀ │p₁₁ │p₁₂│
  ├────┼────┼────┼────┤
  │p₁₃ │p₁₄ │p₁₅ │p₁₆│
  └────┴────┴────┴────┘

  Each patch: 16×16×3 = 768 pixel values → projected to d_vit
```

### Attention inside the ViT is fully bidirectional

Unlike the causal decoder, the ViT uses **no mask** — every patch attends to every other patch in both directions. This lets the encoder build a global understanding of the image before anything reaches the language model.

```
  ViT attention matrix (196 patches, 196×196):

       p₁  p₂  p₃  ...  p₁₉₆
  p₁  [✓   ✓   ✓  ...   ✓  ]   ← p₁ can attend to all patches including future
  p₂  [✓   ✓   ✓  ...   ✓  ]
  p₃  [✓   ✓   ✓  ...   ✓  ]
  ...
  p₁₉₆[✓   ✓   ✓  ...   ✓  ]

  All green — no masking.
  p₁₉₆ can attend to p₁. p₁ can attend to p₁₉₆.
  Global context for every patch.
```

The ViT also prepends a special `[CLS]` token (same idea as BERT). Its output vector after all ViT layers is used as a global image representation for tasks that need a single image embedding (e.g. image classification).

### 4.1 Why 16×16 Patches?

The original ViT paper (Dosovitskiy et al., 2020) tried several patch sizes and measured the accuracy vs compute tradeoff:

| ViT variant | Patch size | Grid (224px input) | Sequence length | Notes |
|---|---|---|---|---|
| ViT-B/32 | 32×32 px | 7×7 | **49 tokens** | Fastest, lowest accuracy |
| ViT-B/16 | 16×16 px | 14×14 | **196 tokens** | Sweet spot — best accuracy/compute |
| ViT-L/14 | 14×14 px | 16×16 | **256 tokens** | Used in CLIP — higher resolution, more tokens |
| ViT-H/14 | 14×14 px | 16×16 | **256 tokens** | Largest ViT variant |

The tradeoff is straightforward:

```
  Smaller patch → more patches → longer sequence → more compute, more detail
  Larger patch  → fewer patches → shorter sequence → cheaper, coarser

  ViT-B/32:  49 patches   — fast, but misses fine detail (text, small objects)
  ViT-B/16: 196 patches   — the standard; 16px captures enough local texture
  ViT-L/14: 256 patches   — worth the cost for CLIP's image-text matching tasks
```

**Why 16px specifically?** There is no deep theoretical reason — it emerged from ablation. 16px patches contain enough local structure (edges, textures, colour gradients) that the subsequent transformer layers can meaningfully relate patches to each other. 32px patches lose too much local detail; 8px patches make the sequence prohibitively long.

**The resolution convention:** Most ViTs are trained at 224×224 pixels. This is inherited from ImageNet benchmarks where all images were resized to 224×224. At inference, different resolutions can be handled by interpolating the positional embeddings (see section 7).

### 4.2 Patch Size vs Grid Size — The "14" Confusion

The number **14** appears in two different ViT contexts with opposite meanings. This is a persistent source of confusion:

| Model name | What "14" means | Patch size | Grid size | Token count |
|---|---|---|---|---|
| CLIP **ViT-L/14** | **14 = patch size in pixels** | 14×14 px | 16×16 patches | 256 |
| **ViT-B/16** | 16 = patch size in pixels | 16×16 px | **14×14 patches** | 196 |

```
  CLIP ViT-L/14 (14px patch size):
  ┌──────────────────────────────────────┐
  │ 224px ÷ 14px = 16 patches per side  │
  │ → 16×16 grid → 256 patch tokens     │
  └──────────────────────────────────────┘

  ViT-B/16 (16px patch size):
  ┌──────────────────────────────────────┐
  │ 224px ÷ 16px = 14 patches per side  │
  │ → 14×14 grid → 196 patch tokens     │
  └──────────────────────────────────────┘
```

The naming convention `ViT-{size}/{patch_px}` names models by their **patch size in pixels**. The grid size (number of patches per side) is 224 ÷ patch_px — so smaller patch size means **larger** grid.

![Patch size vs grid size disambiguation](images/patch_size_vs_grid_size_disambiguation.svg)

> **Rule of thumb:** The number after the slash in a ViT name is always the patch size in **pixels**. The grid size is always 224 ÷ that number (for standard 224px inputs).

---
