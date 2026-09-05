# Foundations and Learning Roadmap

> A bottom-up guide to everything that leads to modern LLMs, VLMs, and VLAs.
> Read this file first. It is the map — all other files are the territory.

---

## The Full Learning Map

```
PHASE 0 — Pre-Transformers & Foundations                 ← 00_Foundations_and_Pre_Transformers/
  ├─ RNN Era & Bahdanau attention
  ├─ Reconstruction family (VAEs, MAE)
  ├─ Generative flows & Diffusion (DDPM, Flow Matching)
  └─ Contrastive family (Word2Vec, CLIP, SigLIP)

PHASE 1 — Transformer Mechanics                          ← 01_Transformer_Core_Mechanics.md
  ├─ Token embeddings, QKV attention, scaling
  ├─ Multi-head attention, FFN, residuals, LayerNorm
  └─ Positional encodings: sinusoidal, ALiBi, RoPE

PHASE 2 — Architectures & Model Families                 ← 02_Architecture_and_Model_Families.md
  ├─ Encoder-decoder (original Transformer, T5)
  ├─ Decoder-only (GPT, LLaMA, Mistral)
  └─ Encoder-only (BERT)

PHASE 3 — Training and Alignment                         ← 03_Training_and_Alignment/
  ├─ Pretraining Objectives (MLM, NSP, Causal)
  ├─ Supervised Fine-Tuning (SFT) & LoRA
  └─ Alignment (RLHF, DPO, GRPO)

PHASE 4 — Decoding and Generation                        ← 04_Decoding_and_Output_Generation.md
  ├─ Temperature scaling
  ├─ Greedy vs Beam Search
  ├─ Sampling (Top-k, Top-p)
  └─ Constrained / Guided Decoding

PHASE 5 — Inference, Hardware, and Scaling               ← 05_Inference_Hardware_and_Scaling/
  ├─ KV cache, GQA/MQA, PagedAttention
  ├─ FlashAttention, Quantization
  ├─ Distributed training (ZeRO/FSDP, pipeline)
  └─ Scaling with Mixture of Experts (MoE)

PHASE 6 — Vision Transformers & Multimodality            ← 06_Multimodality_VLMs_and_VLAs.md
  ├─ ViT (patch tokenisation, CLS)
  ├─ VLMs (ViT + adapter + causal LLM)
  └─ VLAs (action bins, compounding error)

PHASE 7 — Conceptual Q&A                                 ← 07_Conceptual_QA_Bank.md
  └─ Self-assessment for deep understanding
```
