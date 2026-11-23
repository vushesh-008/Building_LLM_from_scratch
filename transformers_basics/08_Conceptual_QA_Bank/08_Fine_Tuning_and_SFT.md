## Fine-Tuning & SFT (Classification vs. Instruction)

---

**Q: In GPT Classification Fine-Tuning, why do we pass only the *last* token's output (`logits[:, -1, :]`) to the loss function instead of the first token like BERT's `[CLS]`?**

GPT uses a **causal (look-ahead) attention mask**, where token $t$ can only attend to tokens at positions $\le t$.
- Position `0` (the first token) only sees itself — it has zero information about the rest of the sentence.
- Position `-1` (the **last token**) is the only position in the sequence that has attended to every single token from position `0` through `T-1`.

In contrast, BERT uses **bidirectional attention** with no causal mask, so `[CLS]` at position `0` can attend to all tokens to its right.

---

**Q: In Instruction Fine-Tuning, why do we keep the *first* `<|endoftext|>` (`50256`) token in `targets` and only replace the *remaining* `<|endoftext|>` padding tokens with `-100`?**

If we replaced **all** `<|endoftext|>` (`50256`) tokens in `targets` with `-100`, the model would never receive a gradient update teaching it to emit `<|endoftext|>` when its answer is complete — at inference time, it would keep generating text forever after answering.

By keeping the **first** `50256` token (right after the final word of the response) unmasked in `targets`, the model learns: *"given the last token of the answer, predict `<|endoftext|>`."* All subsequent `50256` tokens only exist to pad shorter sequences to match the batch length, so they are replaced with `-100` to avoid penalizing the model on artificial padding.

```
  Raw shifted target:  [201,   202,   50256,  50256,  50256]
  Masked target:       [201,   202,   50256,  -100,   -100 ]
                                        ▲       ▲       ▲
                            KEEP (real EOS)     MASK (padding)
```

---

**Q: Why do we put `-100` only in `targets` and NEVER in `inputs`?**

Two reasons:
1. **`nn.Embedding` lookup crash:** The input tensor `inputs` is passed directly into `nn.Embedding(vocab_size, d_model)`, which indexes row numbers `0` to `vocab_size - 1`. `-100` is not a valid row index and immediately raises an `IndexError`.
2. **`ignore_index=-100` is a loss-function feature:** PyTorch's `F.cross_entropy(logits, targets, ignore_index=-100)` only inspects the `targets` tensor and skips any position where `target == -100` (giving it zero loss and zero gradient). Meanwhile, keeping valid token IDs in `inputs` allows instruction tokens to flow through self-attention as Keys and Values so the response tokens can read the prompt context.

---

**Q: Why do we also mask the *instruction* tokens in `targets` with `-100` during Instruction Fine-Tuning?**

In an instruction dataset, the user's prompt (`### Instruction: ... ### Input: ...`) is given as context — the assistant's job is to generate the **Response**, not to predict what question the user is going to ask.

Masking the instruction positions in `targets` with `-100` ensures that 100% of the gradient updates focus on generating high-quality **responses** (given the instruction in `inputs`), rather than wasting model capacity memorizing prompt phrasing.

