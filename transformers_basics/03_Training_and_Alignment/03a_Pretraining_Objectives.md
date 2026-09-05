# Pre-training Objectives

> Extracted from BERT and foundational architectural decisions.

Before post-training alignment, models undergo massive self-supervised pre-training to learn general representations of language. The objective chosen determines the architecture and capability of the resulting model.

## 1. Causal Language Modeling (Next-Token Prediction)

This is the standard for **Decoder-Only** models (GPT, LLaMA, Mistral, Claude).

**Objective:** Predict the probability of the next token given all previous tokens.

$$L = -\sum_i \log P(\text{token}_i \mid \text{token}_{<i})$$

**Mechanism:** 
- The model uses **causal masking** (also called a look-ahead mask) in its self-attention layer. Token $i$ can only attend to tokens $1$ through $i$.
- This objective forces the model to encode all past context into the current hidden state to accurately predict the future.
- By nature, this trains the model to be an autoregressive generator.

## 2. Masked Language Modeling (MLM)

This is the standard for **Encoder-Only** models (BERT, RoBERTa).

**Objective:** Reconstruct masked tokens from a corrupted input sequence using bidirectional context.

**Mechanism (BERT):**
- 15% of tokens are selected at random.
- Of those, 80% are replaced with `[MASK]`, 10% with a random word, and 10% kept unchanged.
- The model uses **bidirectional attention** (no masking) — every token can see every other token in the sequence.
- The output head attempts to predict the original identity of the masked tokens.

$$L_{\text{MLM}} = -\log P(\text{true token} \mid h_{\text{mask}})$$

**Why MLM?** It produces much richer contextual embeddings because the representation for a word integrates information from both its left and right context simultaneously. The downside is that the model cannot generate text autoregressively.

## 3. Next Sentence Prediction (NSP)

Used alongside MLM in the original BERT (but dropped in later models like RoBERTa).

**Objective:** Given two segments A and B, predict whether B followed A in the original document.

**Mechanism:**
- A `[CLS]` token is prepended to the sequence.
- 50% of the time, B is the actual next sentence. 50% of the time, B is a random sentence from another document.
- The final hidden state of the `[CLS]` token is passed through a binary classifier.

$$L_{\text{NSP}} = -\log P(\text{IsNext} \mid h_{\text{CLS}})$$

It was later found that NSP was too easy of a task and removing it (while training MLM for longer) actually improved downstream performance (the RoBERTa conclusion).
