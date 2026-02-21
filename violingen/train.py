"""Training script for ViolinGen.

Usage
-----
    python -m violingen.train [--epochs N] [--seq-len L] [--lr LR]

The script generates synthetic random sequences to demonstrate training.
Replace the ``get_sequences`` function with real MIDI data for best results.
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .data import ViolinVocab, build_training_sequences, random_sequence
from .model import ViolinGen


def get_sequences(vocab: ViolinVocab, num: int = 50, length: int = 64) -> list:
    """Return synthetic random note sequences for demonstration."""
    return [random_sequence(vocab, length) for _ in range(num)]


def train(
    epochs: int = 10,
    seq_len: int = 32,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
) -> ViolinGen:
    """Train ViolinGen on synthetic sequences and return the trained model."""
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = ViolinVocab()
    model = ViolinGen(vocab_size=len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    sequences = get_sequences(vocab)
    pairs = build_training_sequences(sequences, seq_len=seq_len)

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        steps = 0
        for i in range(0, len(pairs) - batch_size + 1, batch_size):
            batch = pairs[i : i + batch_size]
            inp = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
            tgt = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)

            optimizer.zero_grad()
            log_probs, _ = model(inp)
            # log_probs: (batch, seq_len, vocab_size) → reshape for NLLLoss
            loss = criterion(
                log_probs.reshape(-1, len(vocab)),
                tgt.reshape(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ViolinGen")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model = train(epochs=args.epochs, seq_len=args.seq_len, lr=args.lr)
    torch.save(model.state_dict(), "violingen_weights.pt")
    print("Model saved to violingen_weights.pt")


if __name__ == "__main__":
    main()
