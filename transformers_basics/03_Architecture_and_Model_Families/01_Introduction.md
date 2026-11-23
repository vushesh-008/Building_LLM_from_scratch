# Architecture and Model Families

> Sources: *Attention Is All You Need* (Vaswani et al., 2017) | *BERT* (Devlin et al., 2018) | *Build a Large Language Model (From Scratch)* (Sebastian Raschka) | Stanford CME295

Once you have the core mechanics of a single Transformer block (Folder `02`), how do you stack those blocks into complete foundation models? Every Transformer model in existence belongs to one of **three architectural families**:

```
                        The Original 2017 Transformer
                        (Encoder Stack + Decoder Stack)
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
  1. Encoder-Decoder          2. Decoder-Only             3. Encoder-Only
     (Seq2Seq / Translation)     (Autoregressive LLMs)       (Understanding & Embeddings)
     • Bidirectional Encoder     • Causal Self-Attention     • Bidirectional Self-Attention
     • Causal Decoder +          • No Encoder, no            • No Decoder, no Causal Mask
       Cross-Attention             Cross-Attention           • Pre-trained via MLM (+ NSP)
     • T5, BART, Whisper         • GPT, LLaMA, Claude, Qwen  • BERT, RoBERTa, ModernBERT, ViT
```

---

## Folder Reading Order

1. **[02_How_Blocks_Stack__ShapePreserving_Property.md](file:///Users/vusheshbabuadhikari/Learning/git/Building_LLM_from_scratch/transformers_basics/03_Architecture_and_Model_Families/02_How_Blocks_Stack__ShapePreserving_Property.md)** — Why $[N_{\text{seq}} \times d_{\text{model}}] \to [N_{\text{seq}} \times d_{\text{model}}]$ lets us stack 12 to 100+ identical Transformer blocks sequentially.
2. **[03_The_Original_Transformer_Architecture.md](file:///Users/vusheshbabuadhikari/Learning/git/Building_LLM_from_scratch/transformers_basics/03_Architecture_and_Model_Families/03_The_Original_Transformer_Architecture.md)** — The 2017 Encoder-Decoder setup ($N=6$ blocks each) and how Cross-Attention connects them.
3. **[04_Full_EncoderDecoder_Macro_Flow.md](file:///Users/vusheshbabuadhikari/Learning/git/Building_LLM_from_scratch/transformers_basics/03_Architecture_and_Model_Families/04_Full_EncoderDecoder_Macro_Flow.md)** — Why the Encoder stack runs **once** and broadcasts its final $(K_{\text{enc}}, V_{\text{enc}})$ output to every Decoder block.
4. **[05_DecoderOnly_LLMs_vs_EncoderDecoder.md](file:///Users/vusheshbabuadhikari/Learning/git/Building_LLM_from_scratch/transformers_basics/03_Architecture_and_Model_Families/05_DecoderOnly_LLMs_vs_EncoderDecoder.md)** — How GPT, LLaMA, and Claude discard the Encoder and Cross-Attention sublayer to form a 2-sublayer causal Decoder block.
5. **[06_BERT__EncoderOnly_Models.md](file:///Users/vusheshbabuadhikari/Learning/git/Building_LLM_from_scratch/transformers_basics/03_Architecture_and_Model_Families/06_BERT__EncoderOnly_Models.md)** — Complete guide to BERT & Encoder-Only models: why bidirectional attention can't use next-token prediction, why `[CLS]` and `[SEP]` exist, 3-part input embeddings (Token + Position + Segment), MLM (15% / 80-10-10) + NSP routing, and how to adapt a pre-trained Encoder for **Classification, NER, Extractive QA, or Embeddings**.
6. **[07_EncoderOnly_as_Embedding_Models.md](file:///Users/vusheshbabuadhikari/Learning/git/Building_LLM_from_scratch/transformers_basics/03_Architecture_and_Model_Families/07_EncoderOnly_as_Embedding_Models.md)** — Why raw pre-trained BERT has an anisotropic vector space, how Pooling layers (`Mean`, `[CLS]`, `Last-Token EOS`) work, and **Bi-Encoder (Embedding Model) vs. Cross-Encoder (Reranker)**.
