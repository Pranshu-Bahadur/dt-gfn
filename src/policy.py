from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


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
        Teacher-forced log-probabilities for transitions in seq (B×(T-1)),
        masked to zero where the next token is PAD.
        """
        raise NotImplementedError

    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Returns the flow estimates for each position in seq (B×T).
        """
        raise NotImplementedError


class PolicyPaperMLP(PolicyBase):
    """
    Paper-style MLP policy with NO embeddings:
      one-hot(token) -> shared MLP (LeakyReLU) -> {token logits, flow}.
    Applied time-step-wise to seq[:, :-1] during teacher forcing.
    """
    def __init__(
        self,
        vocab_size: int,
        lstm_hidden: int,   # kept for API compat; unused here
        mlp_layers: int,
        mlp_width: int,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)

        # Shared MLP applied on the last dimension (broadcasts over B,T)
        layers = [nn.Linear(self.vocab_size, mlp_width), nn.LeakyReLU()]
        for _ in range(max(0, mlp_layers - 1)):
            layers += [nn.Linear(mlp_width, mlp_width), nn.LeakyReLU()]
        self.shared_mlp = nn.Sequential(*layers)

        # Heads
        self.head_tok  = nn.Linear(mlp_width, self.vocab_size)  # logits for next token
        self.head_flow = nn.Linear(mlp_width, 1)                # scalar flow per position

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: LongTensor (B, T)
        returns:
          logits: (B, T, V)
          flow  : (B, T)
        """
        # One-hot encode tokens (B, T, V) — no embedding table
        x = torch.nn.functional.one_hot(seq, num_classes=self.vocab_size).to(torch.float32)
        h = self.shared_mlp(x)                 # (B, T, W)
        logits = self.head_tok(h)              # (B, T, V)
        flow   = self.head_flow(h).squeeze(-1) # (B, T)
        return logits, flow

    @torch.jit.export
    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced log-probabilities for actual next tokens.
        Shapes:
          seq    : (B, T)
          return : (B, T-1) with zeros where next token == PAD
        """
        B, T = seq.size(0), seq.size(1)
        if T < 2:
            return torch.empty(B, 0, device=seq.device, dtype=torch.float32)

        logits, _ = self.forward(seq[:, :-1])                # (B, T-1, V)
        logp = torch.log_softmax(logits, dim=-1)             # (B, T-1, V)
        next_ids = seq[:, 1:]                                 # (B, T-1)
        gathered = logp.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        mask = (next_ids != self.pad_id).to(gathered.dtype)  # zero-out PAD steps
        return gathered * mask

    @torch.jit.export
    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        _, flow = self.forward(seq)  # (B, T)
        return flow



class PolicyTransformer(PolicyBase):
    """
    Transformer-encoder policy for DT-GFN.

    Token + positional embeddings → Transformer encoder → shared heads:
      – next-token logits  (B × T × V)
      – flow estimate      (B × T)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 2,
        d_ff: int = 256 * 4,
        dropout: float = 0.1,
        max_len: int = 1024,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)

        # Embeddings
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb   = nn.Embedding(self.max_len,   self.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # (B, T, D)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.head_tok  = nn.Linear(self.d_model, self.vocab_size)
        self.head_flow = nn.Linear(self.d_model, 1)

    def _positional(self, T: int, device: torch.device) -> torch.Tensor:
        T = min(T, self.max_len)
        pos_ids = torch.arange(T, device=device)
        return self.pos_emb(pos_ids).unsqueeze(0)  # (1, T, D)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq : (B, T) int64 token IDs
        Returns:
          logits : (B, T, V)
          flow   : (B, T)
        """
        B, T = seq.size(0), seq.size(1)
        x = self.token_emb(seq) + self._positional(T, seq.device)  # (B, T, D)

        # key padding mask: True where PAD so the encoder can ignore it
        kpm = (seq == self.pad_id)  # (B, T)
        h = self.encoder(x, src_key_padding_mask=kpm)  # (B, T, D)

        logits = self.head_tok(h)                  # (B, T, V)
        flow   = self.head_flow(h).squeeze(-1)     # (B, T)
        return logits, flow

    @torch.jit.export
    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced log-probabilities, masked at PAD.
        """
        B, T = seq.size(0), seq.size(1)
        if T < 2:
            return torch.empty(B, 0, device=seq.device, dtype=torch.float32)

        logits, _ = self.forward(seq[:, :-1])         # (B, T-1, V)
        logp = torch.log_softmax(logits, dim=-1)      # (B, T-1, V)
        next_ids = seq[:, 1:]                         # (B, T-1)
        gathered = logp.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        mask = (next_ids != self.pad_id).to(gathered.dtype)
        return gathered * mask

    @torch.jit.export
    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        _, flow = self.forward(seq)
        return flow
