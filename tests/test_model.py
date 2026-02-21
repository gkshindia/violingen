"""Tests for violingen package."""

from __future__ import annotations

import torch
import pytest

from violingen.data import (
    ViolinVocab,
    VIOLIN_MIDI_LOW,
    VIOLIN_MIDI_HIGH,
    random_sequence,
    build_training_sequences,
)
from violingen.model import ViolinGen
from violingen.generate import generate


class TestViolinVocab:
    def setup_method(self):
        self.vocab = ViolinVocab()

    def test_vocab_size(self):
        # notes in range + 3 special tokens (START, END, REST)
        expected = (VIOLIN_MIDI_HIGH - VIOLIN_MIDI_LOW + 1) + 3
        assert len(self.vocab) == expected

    def test_special_tokens_exist(self):
        assert "<START>" in self.vocab.token_to_idx
        assert "<END>" in self.vocab.token_to_idx
        assert "REST" in self.vocab.token_to_idx

    def test_encode_decode_roundtrip(self):
        tokens = ["<START>", "60", "REST", "72", "<END>"]
        indices = self.vocab.encode(tokens)
        recovered = self.vocab.decode(indices)
        assert recovered == tokens

    def test_note_indices_count(self):
        assert len(self.vocab.note_indices()) == VIOLIN_MIDI_HIGH - VIOLIN_MIDI_LOW + 1

    def test_start_end_rest_idx_distinct(self):
        assert self.vocab.start_idx != self.vocab.end_idx
        assert self.vocab.start_idx != self.vocab.rest_idx
        assert self.vocab.end_idx != self.vocab.rest_idx


class TestRandomSequence:
    def test_length(self):
        vocab = ViolinVocab()
        seq = random_sequence(vocab, length=20)
        assert len(seq) == 20

    def test_valid_indices(self):
        vocab = ViolinVocab()
        seq = random_sequence(vocab, length=50)
        valid = set(vocab.note_indices()) | {vocab.rest_idx}
        assert all(idx in valid for idx in seq)


class TestBuildTrainingSequences:
    def test_basic(self):
        vocab = ViolinVocab()
        seq = list(range(50))
        pairs = build_training_sequences([seq], seq_len=10)
        assert len(pairs) == 40  # 50 - 10 pairs
        for inp, tgt in pairs:
            assert len(inp) == 10
            assert len(tgt) == 10

    def test_input_target_offset(self):
        seq = list(range(20))
        pairs = build_training_sequences([seq], seq_len=5)
        inp, tgt = pairs[0]
        assert inp == [0, 1, 2, 3, 4]
        assert tgt == [1, 2, 3, 4, 5]


class TestViolinGen:
    def setup_method(self):
        self.vocab = ViolinVocab()
        self.model = ViolinGen(
            vocab_size=len(self.vocab),
            embed_dim=16,
            hidden_size=32,
            num_layers=2,
        )

    def test_forward_shape(self):
        batch, seq_len = 2, 8
        x = torch.randint(0, len(self.vocab), (batch, seq_len))
        log_probs, hidden = self.model(x)
        assert log_probs.shape == (batch, seq_len, len(self.vocab))

    def test_hidden_shape(self):
        x = torch.randint(0, len(self.vocab), (1, 4))
        _, (h, c) = self.model(x)
        assert h.shape == (self.model.num_layers, 1, self.model.hidden_size)
        assert c.shape == (self.model.num_layers, 1, self.model.hidden_size)

    def test_log_probs_sum_to_one(self):
        x = torch.randint(0, len(self.vocab), (1, 5))
        log_probs, _ = self.model(x)
        probs = torch.exp(log_probs)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_init_hidden(self):
        h, c = self.model.init_hidden(batch_size=3)
        assert h.shape == (self.model.num_layers, 3, self.model.hidden_size)
        assert torch.all(h == 0)
        assert torch.all(c == 0)

    def test_parameter_count_tiny(self):
        total = sum(p.numel() for p in self.model.parameters())
        # Ensure the model is "tiny" (< 200k parameters with default settings)
        assert total < 200_000


class TestGenerate:
    def setup_method(self):
        self.vocab = ViolinVocab()
        self.model = ViolinGen(
            vocab_size=len(self.vocab),
            embed_dim=16,
            hidden_size=32,
            num_layers=1,
        )

    def test_generate_length(self):
        seq = generate(self.model, self.vocab, length=10)
        assert len(seq) <= 10

    def test_generate_valid_tokens(self):
        seq = generate(self.model, self.vocab, length=20)
        valid = set(self.vocab.idx_to_token.values()) - {"<START>"}
        for token in seq:
            assert token in valid

    def test_generate_temperature(self):
        torch.manual_seed(0)
        seq_low = generate(self.model, self.vocab, length=15, temperature=0.1)
        torch.manual_seed(0)
        seq_high = generate(self.model, self.vocab, length=15, temperature=2.0)
        # Both should produce valid tokens; just check they run without error
        assert isinstance(seq_low, list)
        assert isinstance(seq_high, list)
