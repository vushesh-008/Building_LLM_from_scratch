## Encoder-Only & Embedding Models (Text and Vision)

---

**Q: Can we call Encoder-Only models like BERT "Embedding Models"? Why isn't a raw pre-trained BERT a good sentence embedding model out of the box?**

Architecturally, yes — Encoder-Only models use **bidirectional self-attention**, so every token attends to the full sentence simultaneously, making them natural backbones for sequence embeddings.

However, a **raw pre-trained BERT** (trained only on Masked Language Modeling) produces a highly **anisotropic** vector space: sentence vectors cluster inside a narrow cone where unrelated sentences still have high cosine similarity. This happens because MLM only optimizes token-level predictions over a vocabulary head (`d_model → vocab_size`), never constraining the geometry between whole-sentence vectors.

To become a true **Embedding Model** (Bi-Encoder), you must:
1. Add a **pooling layer** (typically **Mean Pooling** over non-padded tokens, `[CLS]` pooling, or **Last-Token `EOS` pooling** if adapting a causal Decoder LLM like `Qwen3-Embedding`) followed by **L2 normalization**.
2. Fine-tune with a **contrastive loss (InfoNCE)** so cosine similarity directly measures semantic similarity.

---

**Q: How is InfoNCE (Multiple Negatives Ranking Loss) mathematically connected to standard Cross-Entropy Loss?**

**InfoNCE is literally Cross-Entropy Loss applied over scaled cosine similarity scores instead of vocabulary logits!**

- In **Decoder SFT**, for each token position you compute a vector of logits over the vocabulary $z \in \mathbb{R}^V$, and pass `(logits, target_token_id)` into `F.cross_entropy`.
- In **InfoNCE**, given a batch of $N$ `(query_i, positive_doc_i)` pairs, you compute the $N \times N$ matrix of temperature-scaled cosine similarities $S_{ij} = \frac{\text{sim}(q_i, d_j)}{\tau}$. For row $i$ (query $i$), the vector of similarity scores $S_{i, :} \in \mathbb{R}^N$ acts as the "logits", and the matching document index $y_i = i$ (on the diagonal) is the "target class":

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\exp(\text{sim}(q_i, d_i^+) / \tau) + \sum_{j \neq i} \exp(\text{sim}(q_i, d_j^-) / \tau)}$$

---

**Q: What is Matryoshka Representation Learning (MRL) Loss, and what problem does it solve?**

Standard 1024-d or 4096-d embeddings spread semantic information evenly across all dimensions — truncating a vector at inference time ruins retrieval accuracy, forcing vector databases to store all dimensions in expensive RAM.

**Matryoshka Loss** (used in `OpenAI text-embedding-3`, `Nomic-Embed`, and `Qwen3-Embedding`) applies **InfoNCE loss simultaneously at multiple nested dimension prefixes** during training (e.g., first `64, 128, 256, 512, 1024` dimensions):

$$\mathcal{L}_{\text{MRL}} = \sum_{m \in \{64, 128, 256, 512, 1024\}} w_m \cdot \mathcal{L}_{\text{InfoNCE}}(e_{[:, :m]})$$

This forces the model to pack the coarsest, highest-variance semantic information into the earliest coordinates, allowing you to slice a 1024-d vector down to 256-d or 128-d at deployment with minimal accuracy loss.

---

**Q: When training a Visual Encoder (ViT) with Masked Autoencoding (MAE), why do we mask 75% of image patches whereas BERT only masks 15% of text tokens?**

**Information density vs. spatial redundancy:**
- **Text (BERT — 15%):** Words are discrete symbols with high information density. Masking 15% of words already removes critical semantic concepts that cannot be guessed by simple linear interpolation of neighboring words.
- **Vision (MAE — 75%):** Pixels and adjacent image patches have massive spatial continuity and redundancy. If you only mask 15% of image patches, the ViT can trivially guess the missing patch's colors from its immediate 8 neighbors without learning high-level object semantics. Masking **75%** destroys local texture-copying shortcuts and forces the ViT encoder to learn holistic object structure. Furthermore, because the MAE encoder only runs self-attention on the **25% unmasked patches**, encoder pre-training is $4\times$ faster!

---

**Q: Why are modern VLMs switching from CLIP (Symmetric InfoNCE) to SigLIP (Pairwise Sigmoid Loss) for their Vision Encoder?**

In **CLIP**, InfoNCE uses a softmax over all $N$ images/captions in the batch: $\sum_{j=1}^N \exp(I_i \cdot T_j / \tau)$. Computing that denominator requires synchronizing (`all_gather`) all embeddings across all GPUs at every step and can be unstable at smaller batch sizes.

**SigLIP** treats every $(I_i, T_j)$ cell in the $N \times N$ matrix as an **independent binary classification problem** ($z_{ij} = +1$ on the diagonal, $z_{ij} = -1$ off-diagonal) using Sigmoid Binary Cross-Entropy. Because there is no batch-wide softmax normalization denominator, memory and distributed communication scale much better, and training remains stable across both small and large batch sizes.

