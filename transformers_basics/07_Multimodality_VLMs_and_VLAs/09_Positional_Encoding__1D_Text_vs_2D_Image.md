## 7. Positional Encoding — 1D Text vs 2D Image

### Text: position = order in sequence

```
  "The  cat  sat  on  the  mat"
    0    1    2    3    4    5     ← 1D position indices

  Position 3 vs 4 matters (word order).
  But the absolute index 3 vs 300 matters less — relative position is key.
```

### Image patches: position = location in 2D grid

```
  14×14 patch grid:

  (0,0) (0,1) (0,2) ... (0,13)   ← top row
  (1,0) (1,1) (1,2) ... (1,13)
  ...
  (13,0)(13,1)(13,2)... (13,13)  ← bottom row

  Patch (0,0) is top-left corner.
  Patch (13,13) is bottom-right corner.
  Swapping them would flip the image — spatial position is semantically critical.
```

![1D positional encoding for text vs 2D for image patches](images/positional_encoding_1d_vs_2d.svg)

### Two approaches to 2D position

**Option 1 — 2D positional encoding (explicit):**

```
  Each patch gets:  row_embedding(r) + col_embedding(c)

  Patch (3, 7):  pos_emb = E_row[3] + E_col[7]

  Separate learned row and column embedding tables.
  Explicitly encodes 2D structure.
```

**Option 2 — Flatten to 1D (implicit):**

```
  Grid:                            Flattened:
  (0,0)(0,1)(0,2)(0,3)            pos 0, 1, 2, 3,
  (1,0)(1,1)(1,2)(1,3)    →       pos 4, 5, 6, 7,
  (2,0)(2,1)(2,2)(2,3)            pos 8, 9, 10, 11,
  (3,0)(3,1)(3,2)(3,3)            pos 12, 13, 14, 15

  Row-major order. Standard 1D sinusoidal or learned PE applied.
  The ViT must learn from the training data that positions 0–3 are a row,
  positions 4–7 are the next row, etc.
  Simpler to implement; most ViT variants use this.
```

Once the patches enter the LLM decoder, they are treated as a flat 1D prefix — the decoder applies its standard 1D positional encoding on top of their already-encoded patch positions.

### What actually happens in practice

Despite the theoretical elegance of 2D encoding, the original ViT ablation showed almost no accuracy difference between 1D and 2D positional encodings. Most ViTs — including the CLIP ViT variants — use **plain 1D learned positional embeddings** (one learned vector per position index, 0 to N−1).

```
  What the ViT paper found:
  ┌─────────────────────────────┬──────────┐
  │ Positional encoding type    │ Accuracy │
  ├─────────────────────────────┼──────────┤
  │ None                        │  61.4%   │
  │ 1D learned (standard)       │  64.0%   │  ← baseline
  │ 2D learned (row + col)      │  64.0%   │  ← same
  │ Relative (learned)          │  64.1%   │  ← marginal
  └─────────────────────────────┴──────────┘

  The transformer appears to learn 2D structure implicitly from 1D positions
  when trained on enough image data.
```

**Inference at a different resolution:** If you train at 224px (196 patches) and run inference on a 336px image (441 patches), the positional embedding table has the wrong size. The fix is **bilinear interpolation** of the learned position vectors to the new grid size. This is standard practice in models like CLIP and LLaVA.

```
  Training:  224px input → 14×14 grid → 196 position vectors learned
  Inference: 336px input → 21×21 grid → 441 positions needed

  Solution: interpolate the 14×14 grid of vectors → resized to 21×21
  (bilinear interpolation in 2D, then flatten back to 1D)
```

---
