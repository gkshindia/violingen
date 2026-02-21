"""Tiny autoregressive violin model.

Architecture: embedding → LSTM → linear projection → log-softmax.
Kept intentionally small so it can be trained and run on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViolinGen(nn.Module):
    """Tiny LSTM-based autoregressive model for generating violin note sequences.

    Parameters
    ----------
    vocab_size:
        Total number of tokens (notes + special tokens).
    embed_dim:
        Dimension of the token embedding.
    hidden_size:
        Number of hidden units in each LSTM layer.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability applied between LSTM layers (ignored when
        ``num_layers == 1``).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x:
            Integer token indices of shape ``(batch, seq_len)``.
        hidden:
            Optional initial LSTM hidden state ``(h_0, c_0)``.

        Returns
        -------
        log_probs:
            Log-probability distribution over tokens, shape
            ``(batch, seq_len, vocab_size)``.
        hidden:
            Updated LSTM hidden state.
        """
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        log_probs = torch.log_softmax(self.fc(out), dim=-1)
        return log_probs, hidden

    def init_hidden(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed initial hidden state for the given batch size."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c
