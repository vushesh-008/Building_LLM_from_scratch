## 1. ViT Pretraining Objectives

Before a ViT is bolted into a VLM, it needs to be pretrained so that its patch representations are meaningful. Three major pretraining strategies exist — each uses a different loss function and different supervision signal.

![Three pretraining objectives for vision transformers](images/vit_pretraining_objectives.svg)

### 1.1 Classification (original ViT)

```
  Supervision:  ImageNet labels  (1000 classes, human annotated)

  Mechanism:
  Image → ViT → [CLS] token vector → Linear head → logits over 1000 classes
                                                          ↓
                                              Cross-entropy vs true label

  Loss:  CE = -log P(correct class)
```

- Requires large annotated datasets (ImageNet-21k for larger ViTs)
- [CLS] token learns a global image summary
- Works well but label collection is expensive and class-centric (not general)

### 1.2 Contrastive (CLIP / SigLIP)

```
  Supervision:  (image, caption) pairs  — scraped from the web, no manual labels

  Mechanism:
  Image  → ViT image encoder  → image embedding  [d]
  Caption → Text encoder       → text embedding   [d]

  Build N×N similarity matrix for a batch of N pairs:
  ┌──────────────────────────────────────────┐
  │        cap₁   cap₂   cap₃  ...  capN    │
  │ img₁  [ ✓     ✗      ✗    ...   ✗  ]   │  ← diagonal = matching pairs
  │ img₂  [ ✗     ✓      ✗    ...   ✗  ]   │
  │ img₃  [ ✗     ✗      ✓    ...   ✗  ]   │
  │  ...                                    │
  │ imgN  [ ✗     ✗      ✗    ...   ✓  ]   │
  └──────────────────────────────────────────┘

  CLIP loss:  InfoNCE — pull matching pairs together, push non-matches apart
  SigLIP loss: Sigmoid binary CE — treats each (img, cap) pair independently
```

| | CLIP (InfoNCE) | SigLIP (sigmoid) |
|---|---|---|
| Loss type | Softmax over batch | Per-pair sigmoid |
| Batch dependency | All negatives in batch used | Independent per pair |
| Stability | Can be unstable at small batch | More stable |
| Label source | No labels needed | No labels needed |

- No human labels required — web-scale data (400M pairs for CLIP)
- Produces embeddings that align vision and language
- The ViT from CLIP is what gets used in most VLMs (LLaVA, InstructBLIP, etc.)

### 1.3 Masked Autoencoder (MAE)

```
  Supervision:  The image itself  — fully self-supervised, no labels at all

  Mechanism:
  1. Randomly mask 75% of patches (replace with [MASK] token)
  2. ViT encoder sees only the 25% visible patches
  3. Lightweight decoder reconstructs the masked patches' raw pixel values

  ┌────┬────┬────┬────┐       ┌────┬────┬────┬────┐
  │ p₁ │ ██ │ p₃ │ ██ │  →   │ p₁ │ p̂₂ │ p₃ │ p̂₄ │
  ├────┼────┼────┼────┤  ViT  ├────┼────┼────┼────┤
  │ ██ │ p₆ │ ██ │ p₈ │  +   │ p̂₅ │ p₆ │ p̂₇ │ p₈ │
  └────┴────┴────┴────┘ dec.  └────┴────┴────┴────┘
  Input (25% visible)          Reconstructed output

  Loss:  MSE = mean((predicted_pixels − true_pixels)²)
         computed only on masked patches (not the visible ones)
```

- 75% masking ratio is intentionally high — forces the model to understand global context, not just copy local patches
- No labels needed; can train on any unlabelled image collection
- MAE representations tend to be stronger for dense prediction tasks (segmentation, detection) than CLIP representations
- The ViT encoder in MAE is the part that gets used downstream; the decoder is discarded after pretraining

### 1.4 Comparison

| | Classification | Contrastive (CLIP) | Masked Autoencoder |
|---|---|---|---|
| Supervision | ImageNet labels | (image, caption) pairs | Raw pixels (self) |
| Loss | Cross-entropy | InfoNCE / sigmoid | MSE on masked patches |
| Labels needed | Yes (expensive) | No (web-scraped) | No |
| Output used | [CLS] token | Image/text embeddings | Patch token embeddings |
| Good for | Classification tasks | Zero-shot, VLMs | Dense tasks, fine-tuning |
| Key model | original ViT | CLIP, SigLIP | MAE |

> **For VLMs:** CLIP-pretrained ViTs are the most common choice because the contrastive objective already aligns vision representations with language — which is exactly what a VLM needs.

---
