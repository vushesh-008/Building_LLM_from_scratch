# Questions & Answers

> Covers material from all transformer curriculum phases. Questions test *why* decisions were made, not just definitions.

---

## Positional Encoding

---

**Q: Why does the sinusoidal PE formula use sin/cos at different frequencies rather than just position indices?**

Plain position indices (0, 1, 2, 3...) would work as a signal, but they have two problems: they grow unboundedly for long sequences, and the model has to learn to interpret raw integers. Sinusoidal functions solve both — they are bounded between –1 and +1 regardless of sequence length, and each dimension oscillates at a different frequency, so every position gets a unique fingerprint across the full embedding vector.

The different frequencies also matter geometrically: low-frequency dimensions change slowly across positions (capturing coarse, long-range structure) while high-frequency dimensions change rapidly (capturing fine, local structure). The model can read off both coarse and fine position information from the same vector.

```
  dim 0 (high freq):  sin(1·m),  sin(2·m),  sin(3·m) ...  changes fast
  dim 512 (low freq): sin(0.001·m) ...                     changes slowly

  → Together they give a unique, stable fingerprint for every position m
```

---

**Q: Why does the dot product of two sinusoidal PEs depend only on their *relative* distance and not absolute positions — and why is that desirable?**

Using $\cos(a - b) = \cos a \cos b + \sin a \sin b$, the dot product of $PE_m$ and $PE_n$ simplifies to a sum of $\cos(\omega_i(m - n))$ terms. The result is a function of $(m - n)$ only — the absolute positions $m$ and $n$ cancel out.

This is desirable because language meaning is usually about *relative* relationships ("the word 3 positions back") rather than absolute positions ("word number 47 in the document"). A model that encodes relative distance generalises better — the pattern "subject is 2 tokens before verb" works whether the subject is at position 5 or position 500.

---

**Q: Why does RoPE rotate Q and K *inside* the attention layer rather than adding to the embeddings like sinusoidal PE?**

Sinusoidal PE adds positional information to the token embeddings before they enter the transformer. This means the positional signal gets mixed into the embedding through every subsequent linear projection — by the time you compute Q and K, the position information is entangled with the semantic content in a way that's hard for the model to cleanly separate.

RoPE applies the rotation directly to Q and K vectors just before the dot product. This keeps the positional information precisely where it matters (in the attention score computation) and out of where it doesn't (the value vectors). Crucially, it makes the dot product $q_m \cdot k_n^T$ a function of the relative position $(n - m)$ exactly, not approximately — which is the theoretical guarantee that makes RoPE superior.

```
  Sinusoidal: v_i* = v_i + p_i   →  position baked into embedding everywhere
  RoPE:       q_m  → q_m · R_m   →  position applied only at attention score step
              k_n  → k_n · R_n
              q_m · k_n^T = f(n - m)  exactly
```

RoPE also generalises better to sequence lengths not seen during training, which is why it is the default in modern models (LLaMA, Mistral, Gemma).

---

**Q: Why does ALiBi add a bias directly to attention *scores* rather than to input embeddings?**

Adding to embeddings changes what the token *is* before any processing happens. ALiBi's goal is narrower: just penalise attention scores between tokens that are far apart, so the model learns to rely more on nearby context.

By adding the bias directly to the pre-softmax attention scores — $\text{softmax}(q \cdot k^T / \sqrt{d_k} + \text{bias}(m,n))$ — ALiBi leaves the token representations completely untouched and targets exactly the right place: the weights that determine how much each token attends to every other. The bias is a simple linear function of distance, learned per head, and adds almost no parameters.

---

## Attention & Multi-Head

---

**Q: Why scale attention scores by $\sqrt{d_k}$? What goes wrong without it?**

Each attention score is a dot product of a query and a key: $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$. If the elements are roughly unit normal, the variance of this sum is $d_k$ — so the standard deviation grows as $\sqrt{d_k}$.

With $d_k = 768$, scores can easily reach magnitudes of ~28 before softmax. Softmax of large values is extremely peaked — one entry gets probability ≈ 1 and all others ≈ 0. In this saturated regime the gradients of softmax are nearly zero (the function is flat), and learning stalls.

Dividing by $\sqrt{d_k}$ brings the variance back to 1, keeping scores in a range where softmax has meaningful gradients:

```
  Without scaling (d_k = 768):  scores ~ N(0, 28)  →  softmax saturates  →  vanishing gradients
  With scaling:                  scores ~ N(0, 1)   →  softmax healthy    →  learning works
```

---

**Q: Why does self-attention have no notion of order without positional encoding?**

The attention score between token $i$ and token $j$ is computed purely from their content vectors: $s_{ij} = q_i \cdot k_j^T$. There is no index, no notion of "before" or "after" — just two vectors and their dot product.

If you shuffle the input sequence, the same attention scores are produced (for the same pairs), just in a different arrangement. The output of attention is a weighted sum of value vectors — that weighted sum is the same regardless of the order the tokens arrived in. Self-attention is, by design, **permutation equivariant**. Positional encoding breaks this symmetry by injecting order information into the vectors before attention sees them.

---

**Q: Why are Q heads never shared in GQA/MQA, even though K and V are?**

The whole purpose of multi-head attention is that different heads attend to different things — one head might track syntactic dependencies, another co-reference, another local context. What makes each head different is its query projection: $Q_i = x \cdot W_{Q_i}$. Each head asks a different question of the sequence.

Sharing Q heads would force all heads to ask the identical question — they'd produce the same attention weights and the same output, making multiple heads pointless.

K and V can be shared because the *information available* to answer queries doesn't need to be multiplied — one set of keys and values contains the full sequence content. Different Q heads can query the same K/V from different angles and still attend to different parts of the sequence.

```
  Share K/V: multiple perspectives on the same information  ✓  (this is the point)
  Share Q:   multiple heads asking the same question        ✗  (defeats the purpose)
```

---

**Q: Why is the FFN per-token but attention is cross-token — and why does that matter for MoE?**

Attention computes relationships between tokens — token $i$'s output depends on tokens $j, k, l...$ via the key-query dot products. You cannot process one token independently of others in the attention layer.

The FFN that follows processes each token's representation completely independently — the same function is applied to token 1, then token 2, then token 3, with no information flowing between them. Token $i$'s FFN output depends only on token $i$'s input.

This independence is what makes MoE possible in the FFN layer. Because each token is processed separately, you can route each token to a different expert without any cross-token communication issues. If you tried to put MoE in the attention layer, different tokens attending to each other through different experts would create a consistency problem — Expert A's keys would need to be compatible with Expert B's queries.

---

## BERT

---

**Q: Why does BERT use 80/10/10 for masking instead of always replacing with [MASK]?**

If 100% of selected tokens were always replaced with [MASK], the model would learn that [MASK] tokens need to be predicted but real tokens don't — at fine-tuning time there are no [MASK] tokens, so the model's representations for real tokens would be undertrained.

The 10% random replacement forces the model to maintain good representations for every token position, because any token could be the "wrong" word that needs to be corrected. The 10% unchanged forces the model to keep the original token's representation useful, because sometimes the "answer" is just the token itself. Together they close the gap between pre-training (lots of [MASK]) and fine-tuning (no [MASK]).

```
  80% → [MASK]:   model must predict from context          (main learning signal)
  10% → random:   model must detect and correct errors     (no free ride on real tokens)
  10% → unchanged: model must copy/confirm the real word   (representation stays useful)
```

---

**Q: Why is [CLS] placed at the *beginning* rather than the end?**

BERT uses bidirectional attention — every token attends to every other token in both directions. So [CLS] sees the full sequence regardless of whether it's at position 0 or position N.

The practical reason is causal: in the original Transformer decoder (and for compatibility with later causal models), processing flows left to right, so putting the classification token at the start means it accumulates information from the full sequence through attention by the time it reaches the final layer. It also makes engineering simpler — the classification head always reads from index 0 of the output, regardless of sequence length.

---

**Q: Why does BERT train on sequences of 128 tokens for 90% of steps, then 512 for only 10%?**

Attention is $O(N^2)$ in sequence length — training on 512-token sequences costs $(512/128)^2 = 16\times$ more compute per step than 128-token sequences.

Most of what BERT needs to learn (word-level semantics, local syntactic patterns, MLM) can be learned efficiently on shorter sequences. The final 10% on 512-token sequences teaches the model to use its positional embeddings at longer ranges and handle longer-range dependencies — but this requires far fewer steps because the shorter-context knowledge already built up. The schedule gives 90% of the learning cheaply, then buys long-context ability efficiently at the end.

---

**Q: Why pre-norm (modern) vs post-norm (original paper)?**

**Post-norm** (original): $\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$

The residual connection adds the raw sublayer output before normalising. At the start of training, sublayer outputs can be large and noisy, and the raw residual compounds this — deep post-norm networks are notoriously hard to train without careful learning rate warmup.

**Pre-norm** (modern): $\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$

The input is normalised *before* the sublayer. The residual path ($x$) is always clean — the unnormalised signal flows directly through the skip connection. This keeps gradients well-behaved at any depth and makes training stable without warmup tricks.

```
  Post-norm: noisy signal → sublayer → add residual → normalize  (hard to train deep)
  Pre-norm:  clean signal → normalize → sublayer → add residual   (stable at depth)
```

Most modern models (LLaMA, GPT-NeoX) use pre-norm + RMS norm, which drops the mean-subtraction step for further efficiency.

---

## Mixture of Experts

---

**Q: What happens if all tokens route to the same expert? Why is this a real problem and how is it addressed?**

If the gating network always assigns the highest score to Expert 1, then:
- Expert 1 gets all the gradient signal and becomes very capable
- Experts 2–N receive no gradient and stay at random initialisation
- You've effectively paid for N experts but get the capacity of 1

This is called **expert collapse** or **load imbalance**. It's a real training failure mode because the gating network starts randomly, and if Expert 1 happens to be slightly better early on, the gate learns to send more tokens there, making Expert 1 even better — a runaway feedback loop.

The standard fix is an **auxiliary load-balancing loss** added during training:

$$L_{aux} = \alpha \sum_{i=1}^{N} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average gate probability for expert $i$. This penalises uneven routing and forces the gate to spread tokens across experts. In practice $\alpha$ is small (e.g. 0.01) so it doesn't dominate the main task loss.

---

**Q: Why does sparse MoE decouple parameter count from compute, and why does that matter at scale?**

In a dense model, every parameter is used for every token — doubling parameters doubles both memory and compute. In sparse MoE with top-k routing:

- **Total parameters** = N experts × FFN size → can scale N freely
- **Active parameters per token** = k experts × FFN size → fixed, regardless of N

```
  Dense FFN (1 expert, d=768, d_ff=3072):
    Parameters:  2.4M     Active per token:  2.4M

  Sparse MoE (8 experts, top-1):
    Parameters:  19.2M    Active per token:  2.4M   ← same compute!
```

This matters enormously at scale because inference cost is determined by active parameters per token, not total parameters. Mixtral 8×7B has ~46B total parameters but runs with the compute of a ~13B dense model — you get the *knowledge capacity* of a 46B model at the *inference speed* of a 13B model.

---

## Decoding

---

**Q: Why is temperature a complete no-op when using greedy decoding?**

Greedy decoding always picks $\arg\max P$ — the single highest-probability token. Temperature rescales the logits before softmax ($\text{logit} / T$), which changes the probability values but never changes their *ranking*. If token A had a higher logit than token B before dividing by $T$, it still has a higher logit after dividing by $T$ (for any $T > 0$).

```
  Logits:  "the"=3.2,  "a"=1.8,  "an"=1.5

  T=0.5:  P = [0.900, 0.055, 0.030]   argmax = "the"
  T=1.0:  P = [0.621, 0.153, 0.113]   argmax = "the"
  T=2.0:  P = [0.396, 0.197, 0.169]   argmax = "the"
```

Temperature only matters when you **sample** from the distribution — because sampling is sensitive to the *shape* of the distribution, not just its argmax.

---

**Q: Why does top-p adapt to the shape of the distribution while top-k doesn't?**

Top-k always keeps exactly $k$ tokens regardless of the distribution's shape. When the model is very confident (one token at 0.95 probability), top-k=50 still forces you to sample from 50 tokens — 49 of which are near-zero garbage. When the model is uncertain (20 tokens each at ~0.05), top-k=3 throws away most valid options.

Top-p keeps the smallest set of tokens whose cumulative probability reaches $p$. When the model is confident, that set is small (maybe 1–2 tokens). When the model is uncertain, that set is large (maybe 20+ tokens). The nucleus automatically contracts and expands with the distribution:

```
  Peaked distribution ("the" → 0.92):
    top-k=50  →  keeps 49 near-zero tokens (bad)
    top-p=0.9 →  nucleus = {"the"} only    (good)

  Flat distribution (20 tokens × 0.05):
    top-k=3   →  discards 17 valid tokens  (bad)
    top-p=0.9 →  nucleus = 18 tokens       (good)
```

---

**Q: Why does beam search still produce generic/repetitive output despite exploring multiple paths?**

Beam search maximises the joint probability of the full sequence — it finds the sequence most likely under the model. But "most likely" under a language model trained on diverse human text often means the safest, most average, most seen-in-training sequence. It's the mode of the distribution.

Human language is not mode-seeking — people naturally produce diverse, creative, non-generic text. By explicitly hunting for the highest-probability path, beam search systematically avoids anything surprising or diverse. It also tends to repeat phrases, because repetition has high conditional probability given what was just said.

Sampling-based methods avoid this by not optimising for the mode — they draw from the distribution and so naturally produce more varied output.

---

## KV Cache & Memory

---

**Q: Why does training not need a KV cache but inference does?**

During training the full sequence is available at once. All token positions are processed in parallel with causal masking — position 5 can attend to positions 1–4 via the mask without needing to store and retrieve their K/V from a cache. Everything is computed in one forward pass.

During inference the model generates one token at a time, autoregressively. To predict token $t$, it needs K and V for all previous tokens $1$ to $t-1$. Without caching, those K and V vectors would need to be recomputed from scratch at every step — O(N²) total work. The KV cache stores them once and reuses them.

---

**Q: Why does contiguous memory allocation force a full deep copy when beam search forks?**

When beam search creates two branches from the same prompt, both branches will append *different* future tokens to their KV buffers. A single contiguous array cannot be shared between them — if Branch A writes token "left" at position 10, it overwrites whatever Branch B would have written there.

So the server must physically copy the entire prompt's KV tensor into two separate contiguous memory regions before either branch can safely write its next token. With k=3 beams, that's 3 full copies of the prompt KV at fork time — even though the data is identical.

PagedAttention solves this with copy-on-write: all beams share the same physical blocks until one needs to *write* to a shared block, at which point only that one block is copied, not the entire sequence.

---

**Q: Why does PagedAttention's copy-on-write only ever touch the *last partial block* and not earlier shared blocks?**

Once a block is completely full (all $B$ token slots written), it will never be written to again — there are no free slots. A completely full shared block can be read by all beams simultaneously with no risk of any beam overwriting it.

The only block that ever needs a write is the *current* block — the one where new tokens are being appended. At the moment beams diverge, only this last (potentially partial) block is in danger of being written to differently by different beams. Earlier blocks are sealed.

```
  Blocks #1, #2, #3: full (4/4 tokens) → read-only, safe to share forever
  Block #4: partial (2/4 tokens used)  → beams will write different tokens here
             → only this one gets copy-on-write at divergence
```

---

**Q: With GQA (G groups), what is the KV cache memory reduction relative to MHA?**

In MHA with $h$ heads, you cache $h$ K-head vectors and $h$ V-head vectors per token per layer.
In GQA with $G$ groups, you cache $G$ K-head vectors and $G$ V-head vectors.

Reduction factor: $h / G$

```
  h = 32 heads (typical large model):

  G = 32 (MHA):   32 K + 32 V heads cached  →  1× (baseline)
  G = 8  (GQA):    8 K +  8 V heads cached  →  4× smaller KV cache
  G = 1  (MQA):    1 K +  1 V head  cached  →  32× smaller KV cache
```

LLaMA 3 70B uses $h=64$ query heads, $G=8$ KV groups → 8× KV cache reduction vs full MHA, with quality very close to MHA.

---

## VLM / VLA

---

**Q: Why do image patch tokens get `label = –100` during VLA training if they still participate in attention?**

The two roles are independent. As **keys and values** in attention, image tokens are always active — every later token (including action tokens) can attend to them, read their content, and use that information to compute its output. That information flow is essential and happens regardless of labels.

As **prediction targets**, they would be asking the model to reconstruct the input image from scratch — a task the model was never designed to do and that adds no useful gradient signal for the robot control task. Setting their labels to –100 means `CrossEntropyLoss` ignores those positions, contributing zero to the gradient. They inform without being trained on.

```
  Image tokens as Keys/Values:  always active, carry visual info to all later positions  ✓
  Image tokens as Labels:       -100 → zero gradient, not a prediction target           ✗
```

---

**Q: Why does compounding error happen at inference but not during training?**

During training, every timestep in the demonstration trajectory is conditioned on the **ground-truth** previous state — the actual camera frame the human demonstrator was looking at, and the actual arm position at that moment. Even if the model would have made a wrong prediction at step 3, step 4 still receives the correct step-3 image as input. Errors never propagate.

At inference, the model conditions on **its own previous outputs**. A slightly wrong action bin at step 1 moves the arm to a slightly wrong position, which changes what the camera sees at step 2. The model now sees a camera frame it has likely never seen in training (it's off the demonstrated trajectory), and predicts the next action from an unfamiliar state. That prediction is slightly more wrong, moving the arm further off course — and so on.

```
  Training:   [GT image_t1] → action_t1 → [GT image_t2] → action_t2   (errors reset)
  Inference:  [image_t1]    → action_t1 → [wrong image_t2] → worse action_t2 → ...
```

This gap between training distribution and inference distribution is called **covariate shift**, and it's the fundamental challenge of behavioural cloning.

---

**Q: Why use discrete action bins rather than directly regressing continuous joint values?**

The standard transformer output is a probability distribution over a discrete vocabulary — that is the mechanism it was designed for and pre-trained on. Directly regressing a continuous value would require replacing the softmax head with a regression head (e.g. predicting a float with MSE loss), which:

1. Loses the pre-trained output head and requires training a new one from scratch
2. Changes the loss function (MSE vs cross-entropy) in ways that interact poorly with the rest of the fine-tuning
3. Discards the model's uncertainty representation — a probability distribution over bins naturally expresses "I'm not sure if the arm should go left or right" in a way a single float cannot

By binning, the VLA reuses the *exact same* prediction mechanism as language — a softmax over a vocabulary. The only change is that some vocabulary entries now mean "move 8mm in the x-direction" instead of "the word 'cat'." This lets the model transfer all of its pre-trained next-token prediction machinery directly to action prediction.

---

**Q: Why does VLA training need significantly more data than a similarly-sized text fine-tune?**

Two reasons compound each other:

**1. Supervision sparsity.** Only ~3% of tokens in a VLA training example get a gradient — the action bins. The image patches (often 196 tokens per frame) and state/instruction tokens are context only. A 10-step trajectory might have ~2170 tokens total but only 70 supervised tokens. The model is doing enormous amounts of "reading" for a very sparse reward signal.

**2. Physical diversity.** Text fine-tuning can generalise from limited examples because language has regular structure — patterns seen in one context transfer to others. Physical manipulation does not generalise as readily — a policy trained on mugs in one lighting condition, on one table surface, at one camera angle, may fail completely under minor perturbations. Every new object, background, or lighting condition is effectively a new distribution that needs to be covered by training data.

Combined: the model needs to see enough diverse trajectories that both the action token supervision accumulates meaningfully AND the image-state distribution is covered broadly enough to avoid the compounding error problem at inference.

---
