# violingen

A tiny autogenerative violin model built with PyTorch.

## Overview

`violingen` uses a small LSTM-based autoregressive architecture to generate
sequences of violin notes (MIDI range G3–G7). The model learns to predict the
next note given a context window of previous notes.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Training

```bash
python -m violingen.train --epochs 20 --seq-len 32 --lr 1e-3
# saves weights to violingen_weights.pt
```

### Generation

```bash
python -m violingen.generate --length 32 --temperature 1.0 --weights violingen_weights.pt
```

Output is a space-separated list of MIDI note numbers (or `REST`).

### Python API

```python
from violingen import ViolinGen, ViolinVocab
from violingen.generate import generate

vocab = ViolinVocab()
model = ViolinGen(vocab_size=len(vocab))  # tiny model, ~100k params
# ... train or load weights ...
sequence = generate(model, vocab, length=32)
print(sequence)  # e.g. ['60', '62', 'REST', '64', ...]
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Model Architecture

| Component | Details |
|-----------|---------|
| Embedding | 64-dim token embedding |
| Recurrent | 2-layer LSTM, hidden size 128 |
| Output | Linear → log-softmax over vocabulary |
| Vocabulary | 52 tokens (49 notes + REST + START + END) |