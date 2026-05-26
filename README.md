# Whisper Trainer

A lightweight pipeline for fine-tuning OpenAI's [Whisper](https://github.com/openai/whisper) speech-to-text models on custom datasets. Build a dataset, then train — two scripts, no framework.

## What's inside

- **`build_dataset.py`** — assembles a Whisper-compatible dataset from your raw audio + transcripts.
- **`train_whisper.py`** — fine-tunes a Whisper checkpoint on the prepared dataset.
- **`Instructions - Whisper Dataset Builder.pdf`** — step-by-step guide for building the dataset.
- **`Instructions - Whisper Trainer.pdf`** — step-by-step guide for training.

## Quick start

```bash
pip install -r requirements.txt

# 1. Build dataset
python build_dataset.py

# 2. Train
python train_whisper.py
```

Follow the bundled PDF guides for parameter tuning, dataset format, and hardware recommendations.

## Use cases

- Domain-specific transcription (medical, legal, technical jargon)
- Accent / dialect adaptation
- Custom-vocabulary keyword spotting
- Multilingual / code-switching fine-tunes

## Requirements

GPU strongly recommended (CUDA-capable). Exact dependency versions are pinned in `requirements.txt`.

## License

Shared as a reference implementation. Whisper itself is MIT-licensed by OpenAI.
