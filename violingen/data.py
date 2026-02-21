"""Vocabulary and data utilities for violin note sequences.

The violin's standard MIDI range is approximately MIDI 55 (G3) to MIDI 103 (G7).
We add a special REST token and a START/END token to the vocabulary.
"""

from __future__ import annotations

import random
from typing import List, Sequence

# Violin MIDI note range (G3 = 55 to G7 = 103)
VIOLIN_MIDI_LOW = 55
VIOLIN_MIDI_HIGH = 103

# Special tokens
REST_TOKEN = "REST"
START_TOKEN = "<START>"
END_TOKEN = "<END>"


class ViolinVocab:
    """Maps violin MIDI notes and special tokens to integer indices."""

    def __init__(self) -> None:
        notes = [str(n) for n in range(VIOLIN_MIDI_LOW, VIOLIN_MIDI_HIGH + 1)]
        all_tokens = [START_TOKEN, END_TOKEN, REST_TOKEN] + notes
        self.token_to_idx: dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.idx_to_token: dict[int, str] = {i: t for t, i in self.token_to_idx.items()}

    def __len__(self) -> int:
        return len(self.token_to_idx)

    @property
    def start_idx(self) -> int:
        return self.token_to_idx[START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.token_to_idx[END_TOKEN]

    @property
    def rest_idx(self) -> int:
        return self.token_to_idx[REST_TOKEN]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        """Convert a list of token strings to integer indices."""
        return [self.token_to_idx[t] for t in tokens]

    def decode(self, indices: Sequence[int]) -> List[str]:
        """Convert a list of integer indices back to token strings."""
        return [self.idx_to_token[i] for i in indices]

    def note_indices(self) -> List[int]:
        """Return indices corresponding to playable notes (excluding special tokens)."""
        return [
            self.token_to_idx[str(n)]
            for n in range(VIOLIN_MIDI_LOW, VIOLIN_MIDI_HIGH + 1)
        ]


def random_sequence(vocab: ViolinVocab, length: int = 16) -> List[int]:
    """Generate a random sequence of note indices for testing/demo purposes."""
    playable = vocab.note_indices() + [vocab.rest_idx]
    return [random.choice(playable) for _ in range(length)]


def build_training_sequences(
    sequences: List[List[int]], seq_len: int = 32
) -> List[tuple[List[int], List[int]]]:
    """Slice sequences into (input, target) pairs for teacher-forced training.

    Each pair consists of a window of ``seq_len`` tokens as input and the
    same window shifted by one position as the target.
    """
    pairs: List[tuple[List[int], List[int]]] = []
    for seq in sequences:
        for start in range(len(seq) - seq_len):
            inp = seq[start : start + seq_len]
            tgt = seq[start + 1 : start + seq_len + 1]
            pairs.append((inp, tgt))
    return pairs
