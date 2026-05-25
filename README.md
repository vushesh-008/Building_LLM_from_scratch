# Building an LLM from Scratch

A hands-on implementation of a large language model built from the ground up, following **"Build a Large Language Model (From Scratch)"** by [Sebastian Raschka](https://sebastianraschka.com/).

## About

This repository contains my personal implementations, notebooks, and experiments as I work through the book chapter by chapter. The goal is to deeply understand how LLMs work by building one from scratch using PyTorch — no shortcuts.

## Project Structure

```
Building_LLM_from_scratch/
├── data/               # Raw datasets (verdict.txt, etc.)
│   └── download.py     # Script to download datasets
├── notebooks/          # Chapter-by-chapter Jupyter notebooks
│   ├── 01_tokenization.ipynb
│   ├── 02_Data_Loading.ipynb
│   └── 03_Self_Attention.ipynb
├── src/                # Reusable Python modules
│   ├── 01_DataLoading.py
│   ├── 02_Tokenizer_bytepair.py
│   └── 03_Attention.py
├── checkpoints/        # Saved model weights
└── pyproject.toml      # Project dependencies (managed with uv)
```

## Topics Covered

- [x] Tokenization (character-level and BPE)
- [x] Data loading and batching
- [x] Self-attention mechanism
- [ ] Multi-head attention
- [ ] GPT model architecture
- [ ] Pretraining
- [ ] Finetuning

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Register the Jupyter kernel
uv run python -m ipykernel install --user --name building-llm --display-name "Building LLM from Scratch"

# Launch Jupyter
uv run jupyter notebook
```

Select the **"Building LLM from Scratch"** kernel when opening notebooks.

## Download Data

```bash
uv run python data/download.py
```

## Reference

- Book: [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) — Sebastian Raschka
- Original code: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
