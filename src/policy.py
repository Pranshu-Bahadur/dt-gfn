from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class PolicyBase(nn.Module):
    """
    Abstract base class for policy networks.
    """
    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of token-ID sequences (B×T), returns:
          - logits over next token (B×T×V)
          - flow values (B×T)
        """
        raise NotImplementedError

    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Returns log-probabilities for the transitions in seq (B×(T-1)).
        """
        raise NotImplementedError

    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Returns the flow estimates for each position in seq (B×T).
        """
        raise NotImplementedError


class PolicyPaperMLP(PolicyBase):
    """
    LSTM + deep MLP heads policy network from the DT-GFN paper.
    """
    def __init__(
        self,
        vocab_size: int,
        lstm_hidden: int,
        mlp_layers: int,
        mlp_width: int,
    ):
        super().__init__()
        # token embedding → LSTM
        self.embedding = nn.Embedding(vocab_size, lstm_hidden)
        self.rnn = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # shared MLP on top of every time step
        layers: list[nn.Module] = [nn.Linear(lstm_hidden, mlp_width), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.shared_mlp = nn.Sequential(*layers)

        # two heads: next-token logits and flow scalar
        self.head_tok = nn.Linear(mlp_width, vocab_size)
        self.head_flow = nn.Linear(mlp_width, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # seq: (B, T)
        emb, _ = self.rnn(self.embedding(seq))               # (B, T, H)
        h = self.shared_mlp(emb)                              # (B, T, W)
        logits = self.head_tok(h)                             # (B, T, V)
        flow   = self.head_flow(h).squeeze(-1)                # (B, T)
        return logits, flow

    @torch.jit.export
    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T)
        if seq.size(1) < 2:
            return torch.empty(seq.size(0), 0, device=seq.device)
        logits, _ = self.forward(seq[:, :-1])                 # (B, T-1, V)
        logp = logits.log_softmax(dim=-1)
        # gather log-probs of the actual next tokens
        return torch.gather(
            logp, -1, seq[:, 1:].unsqueeze(-1)
        ).squeeze(-1)                                         # (B, T-1)

    @torch.jit.export
    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T)
        _, flow = self.forward(seq)
        return flow                                           # (B, T)
