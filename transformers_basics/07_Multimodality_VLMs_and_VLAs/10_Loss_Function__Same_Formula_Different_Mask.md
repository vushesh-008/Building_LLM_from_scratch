## 8. Loss Function — Same Formula, Different Mask

The cross-entropy formula is identical to a plain LLM:

$$L = -\sum_i \log P(\text{token}_i \mid \text{context})$$

What changes is **which positions $i$ contribute** to the sum.

### Position-by-position breakdown

```
  Token sequence in decoder:

  Position:  1    2  ...  196   197    198   199  200    201   202
  Token:    [p₁][p₂]...[p₁₉₆]["What"]["is"]["in"]["the"]["A"]["cat"]
  Label:    -100 -100... -100   -100  -100  -100  -100   "A"  "cat"
                                                           ↑
                                            loss computed here onward
```

- Image patches: `label = -100` → no loss, no gradient
- Prompt text: `label = -100` → no loss, no gradient (it's given, not generated)
- Response tokens: real label → cross-entropy loss computed

The image tokens and prompt still flow through the network fully — they are keys and values in attention for every later position. They inform without being trained on.

### Teacher forcing during training

At training time, ground-truth previous tokens are always fed as input, regardless of what the model would have predicted:

```
  Ground truth response: "A cat sitting on a mat"

  Training step (teacher forcing):
  Input:   [image tokens] [prompt] "A"     → predict "cat"    ← fed GT "A", not model's output
  Input:   [image tokens] [prompt] "A cat" → predict "sitting" ← fed GT "A cat"
  ...

  Model never sees its own mistakes during training.
  (Same covariate shift problem as VLAs — just less physically dangerous.)
```

---
