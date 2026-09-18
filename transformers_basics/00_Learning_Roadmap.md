# Foundations and Learning Roadmap

> A bottom-up guide to everything that leads to modern LLMs, VLMs, and VLAs.
> Read this file first. It is the map — all other files are the territory.

---

## The Full Learning Map

```
PHASE 0 — Deep Learning Foundations                      ← 00_Deep_Learning_Foundations/
  ├─ Latent Spaces and Autoencoders
  └─ Generative Models (Diffusion, GANs)

PHASE 1 — NLP and Sequence Foundations                   ← 01_NLP_and_Sequence_Foundations/
  ├─ Tokenization & BPE (Text to Integers)
  ├─ Embeddings & Word2Vec (Integers to Vectors)
  └─ RNNs, Seq2Seq Bottleneck, and Attention

PHASE 2 — Transformer Mechanics                          ← 02_Transformer_Core_Mechanics.md
  ├─ Token embeddings, QKV attention, scaling
  ├─ Multi-head attention, FFN, residuals, LayerNorm
  └─ Positional encodings: sinusoidal, ALiBi, RoPE

PHASE 3 — Architectures & Model Families                 ← 03_Architecture_and_Model_Families.md
  ├─ Encoder-decoder (original Transformer, T5)
  ├─ Decoder-only (GPT, LLaMA, Mistral)
  └─ Encoder-only (BERT)

PHASE 4 — Training and Alignment                         ← 04_Training_and_Alignment/
  ├─ Pretraining Objectives (MLM, NSP, Causal)
  ├─ Supervised Fine-Tuning (SFT) & LoRA
  └─ Alignment (RLHF, DPO, GRPO)

PHASE 5 — Decoding and Generation                        ← 05_Decoding_and_Output_Generation.md
  ├─ Temperature scaling
  ├─ Greedy vs Beam Search
  ├─ Sampling (Top-k, Top-p)
  └─ Constrained / Guided Decoding

PHASE 6 — Inference, Hardware, and Scaling               ← 06_Inference_Hardware_and_Scaling/
  ├─ KV cache, GQA/MQA, PagedAttention
  ├─ FlashAttention, Quantization
  ├─ Distributed training (ZeRO/FSDP, pipeline)
  └─ Scaling with Mixture of Experts (MoE)

PHASE 7 — Vision Transformers & Multimodality            ← 07_Multimodality_VLMs_and_VLAs.md
  ├─ ViT (patch tokenisation, CLS)
  ├─ VLMs (ViT + adapter + causal LLM)
  └─ VLAs (action bins, compounding error)

PHASE 8 — Conceptual Q&A                                 ← 08_Conceptual_QA_Bank.md
  └─ Self-assessment for deep understanding
```
