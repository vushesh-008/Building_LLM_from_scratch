## Table of Contents

1. [ViT Pretraining Objectives](#1-vit-pretraining-objectives)
2. [What Changes, What Stays the Same](#2-what-changes-what-stays-the-same)
3. [Text Embedding vs Patch Embedding](#3-text-embedding-vs-patch-embedding)
4. [The Vision Encoder — Inside the ViT](#4-the-vision-encoder--inside-the-vit)
   - 4.1 [Why 16×16 Patches?](#41-why-1616-patches)
   - 4.2 [Patch Size vs Grid Size — The "14" Confusion](#42-patch-size-vs-grid-size--the-14-confusion)
5. [The Adapter / Projection Layer](#5-the-adapter--projection-layer)
6. [Two Attention Regimes in One Model](#6-two-attention-regimes-in-one-model)
7. [Positional Encoding — 1D Text vs 2D Image](#7-positional-encoding--1d-text-vs-2d-image)
8. [Loss Function — Same Formula, Different Mask](#8-loss-function--same-formula-different-mask)
9. [Full Pipeline End-to-End](#9-full-pipeline-end-to-end)
10. [The Big Picture — LLM → VLM → VLA](#10-the-big-picture--llm--vlm--vla)
11. [How a Continuous Number Becomes a Token](#11-how-a-continuous-number-becomes-a-token)
12. [Training Data — What a Training Example Looks Like](#12-training-data--what-a-training-example-looks-like)
13. [Loss Masking — What Actually Gets a Gradient](#13-loss-masking--what-actually-gets-a-gradient)
14. [The Compounding Error Problem at Inference](#14-the-compounding-error-problem-at-inference)

---
