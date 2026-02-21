
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
