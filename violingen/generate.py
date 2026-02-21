"""Generation script for ViolinGen.

Usage
-----
    python -m violingen.generate [--length N] [--temperature T] [--weights PATH]

Generates a sequence of violin MIDI notes and prints them to stdout.
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from .data import ViolinVocab
from .model import ViolinGen


def generate(
    model: ViolinGen,
    vocab: ViolinVocab,
    length: int = 32,
    temperature: float = 1.0,
    seed_token: str | None = None,
    device: torch.device | None = None,
) -> List[str]:
    """Generate a sequence of violin tokens using the trained model.

    Parameters
    ----------
    model:
        Trained ViolinGen model.
    vocab:
        Vocabulary used during training.
    length:
        Number of tokens to generate.
    temperature:
        Sampling temperature. Values < 1 make the distribution sharper
        (more deterministic); values > 1 increase randomness.
    seed_token:
        Optional starting token string. Defaults to ``<START>``.
    device:
        Device to run generation on. Defaults to the model's device.

    Returns
    -------
    List of generated token strings (excluding the seed token).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if seed_token is None:
        seed_token = "<START>"

    token_idx = vocab.token_to_idx.get(seed_token, vocab.start_idx)
    x = torch.tensor([[token_idx]], dtype=torch.long, device=device)
    hidden = model.init_hidden(batch_size=1, device=device)

    generated: List[str] = []
    with torch.no_grad():
        for _ in range(length):
            log_probs, hidden = model(x, hidden)
            # log_probs: (1, 1, vocab_size)
            logits = log_probs[0, 0] / temperature
            probs = torch.exp(logits)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            token = vocab.idx_to_token[int(next_idx)]
            if token == "<END>":
                break
            if token == "<START>":
                x = torch.tensor([[next_idx]], dtype=torch.long, device=device)
                continue
            generated.append(token)
            x = torch.tensor([[next_idx]], dtype=torch.long, device=device)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate violin notes with ViolinGen")
    parser.add_argument("--length", type=int, default=32, help="Number of notes to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--weights", type=str, default=None, help="Path to saved model weights")
    args = parser.parse_args()

    vocab = ViolinVocab()
    model = ViolinGen(vocab_size=len(vocab))

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        print(f"Loaded weights from {args.weights}")

    sequence = generate(model, vocab, length=args.length, temperature=args.temperature)
    print("Generated sequence:")
    print(" ".join(sequence))


if __name__ == "__main__":
    main()
