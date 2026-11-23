# Pre-training Objectives and Loss Foundations

> Extracted from BERT, GPT, and foundational architectural decisions.

Before post-training alignment, models undergo massive self-supervised pre-training to learn general representations of language. The objective chosen determines the architecture and capability of the resulting model.

---

## 0. The Core Loss Function Across All Transformer Families

Whether you are training an **Encoder-Only** model (BERT), an **Encoder-Decoder** model (Original Transformer, T5, BART), or a **Decoder-Only** model (GPT, LLaMA, Mistral, Claude), the fundamental training objective is built on **Cross-Entropy Loss**:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid \text{context}_i)$$

where $y_i$ is the ground-truth target token (or target class) at position $i$, and $P(y_i \mid \text{context}_i)$ is the softmax probability assigned by the model to that ground-truth index.

### Derived Metrics & Task Extensions
- **Perplexity ($\text{PPL}$):** Directly derived from cross-entropy loss by exponentiation:
  $$\text{PPL} = \exp(\mathcal{L}_{\text{CE}}) = e^{\mathcal{L}_{\text{CE}}}$$
  Perplexity measures how "uncertain" the model is when predicting the next token — a perplexity of $k$ intuitively means the model is as confused as if it had to choose uniformly at random among $k$ possible tokens at each step. Lower cross-entropy $\iff$ lower perplexity.
- **Task-Specific & Derived Loss Extensions:** For specialized tasks such as summarization, machine translation, or preference alignment, extensions are often layered on top of (or derived from) cross-entropy — such as **label smoothing**, **sequence-level / RL-based rewards** (optimizing ROUGE/BLEU), **contrastive losses**, or **preference losses (DPO/RLHF)**. However, standard **token-level Cross-Entropy Loss** remains the core engine for pre-training, classification fine-tuning, and instruction fine-tuning. *(We can expand on task-specific derived losses later as extensions.)*

---

## 1. Causal Language Modeling (Next-Token Prediction)

This is the standard for **Decoder-Only** models (GPT, LLaMA, Mistral, Claude) and the decoder side of **Encoder-Decoder** models.

**Objective:** Predict the probability of the next token given all previous tokens.

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(\text{token}_t \mid \text{token}_{<t})$$

**Mechanism:** 
- The model uses **causal masking** (also called a look-ahead mask) in its self-attention layer. Token $t$ can only attend to tokens $1$ through $t$.
- This objective forces the model to encode all past context into the current hidden state to accurately predict the future.
- By nature, this trains the model to be an autoregressive generator.

### Input vs. Shifted Target Tensors in Pre-training
During pre-training (as in *Build a Large Language Model from Scratch*, Ch. 5), a single sequence of length $T+1$ yields $T$ parallel next-token prediction tasks by shifting the target tensor left by **1 position**:

```
  Raw token sequence:  ["Effort", "moves", "mountains", "every", "day"]
                       ─────────────────────────────────────────────────
  Input  (x):          ["Effort", "moves", "mountains", "every"]   shape: [B, T]
  Target (y):          ["moves",  "mountains", "every", "day"  ]   shape: [B, T]

  4 predictions computed in a single forward pass:
    Position 0: given ["Effort"]                                → predict "moves"
    Position 1: given ["Effort", "moves"]                       → predict "mountains"
    Position 2: given ["Effort", "moves", "mountains"]          → predict "every"
    Position 3: given ["Effort", "moves", "mountains", "every"] → predict "day"
```

- **Logits shape:** `[batch_size, seq_len, vocab_size]` (e.g., `[B, T, 50257]`)
- **Targets shape:** `[batch_size, seq_len]` (e.g., `[B, T]`)
- To compute PyTorch's `F.cross_entropy(logits.flatten(0, 1), targets.flatten())`, the first two dimensions are flattened so every token position in every batch item contributes equally to the average cross-entropy loss.

---

## 2. Masked Language Modeling (MLM)

This is the standard for **Encoder-Only** models (BERT, RoBERTa).

**Objective:** Reconstruct masked tokens from a corrupted input sequence using bidirectional context.

**Mechanism (BERT):**
- 15% of tokens are selected at random.
- Of those, 80% are replaced with `[MASK]`, 10% with a random word, and 10% kept unchanged.
- The model uses **bidirectional attention** (no causal masking) — every token can see every other token in the sequence.
- The output head attempts to predict the original identity of the masked tokens using Cross-Entropy Loss only at the 15% masked positions (unmasked positions are ignored via loss masking).

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(\text{true token}_i \mid h_i)$$

**Why MLM?** It produces much richer contextual embeddings because the representation for a word integrates information from both its left and right context simultaneously. The downside is that the model cannot generate text autoregressively.

---

## 3. Next Sentence Prediction (NSP)

Used alongside MLM in the original BERT (but dropped in later models like RoBERTa).

**Objective:** Given two segments A and B, predict whether B followed A in the original document.

**Mechanism:**
- A `[CLS]` token is prepended to the sequence.
- 50% of the time, B is the actual next sentence. 50% of the time, B is a random sentence from another document.
- The final hidden state of the `[CLS]` token is passed through a 2-class linear head and trained with binary Cross-Entropy Loss:

$$\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} \mid h_{\text{CLS}})$$

It was later found that NSP was too easy of a task and removing it (while training MLM for longer) actually improved downstream performance (the RoBERTa conclusion).
