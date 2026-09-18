## 6. Segment Embeddings

BERT needs to distinguish Sentence A from Sentence B within a single input sequence. It does this by adding a learned **segment embedding** to each token:

$$\text{Final input vector} = \text{Token embedding} + \text{Positional encoding} + \text{Segment encoding}$$

Two trainable vectors are learned during pre-training: $E_A$ (segment A) and $E_B$ (segment B).

### Example: "I love cats" + "They are fluffy"

Tokenized: `[CLS] I love cats [SEP] They are fluffy [SEP]`

| Position | Token | Segment | Embedding added |
|---|---|---|---|
| 0 | [CLS] | A | $E_A$ |
| 1 | I | A | $E_A$ |
| 2 | love | A | $E_A$ |
| 3 | cats | A | $E_A$ |
| 4 | [SEP] (middle) | A | $E_A$ |
| 5 | They | B | $E_B$ |
| 6 | are | B | $E_B$ |
| 7 | fluffy | B | $E_B$ |
| 8 | [SEP] (final) | B | $E_B$ |

> Everything from `[CLS]` through the middle `[SEP]` = Segment A. Everything from the first token of Sentence B through the final `[SEP]` = Segment B.

In PyTorch/HuggingFace: Segment A = token_type_id `0`, Segment B = `1`.

---
