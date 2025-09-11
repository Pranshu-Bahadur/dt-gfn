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
        masked to zero where the next token is PAD, and with the single step
        that predicts EOS removed (sampler appends EOS deterministically).
        """
        raise NotImplementedError

    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Returns the flow estimates for each position in seq (B×T).
        """
        raise NotImplementedError


class PolicyPaperMLP(PolicyBase):
    """
    Decayed-prefix readout (retention-like without Q/K/V):

      z_t = Embedding(x_t) ∈ R^H
      S_t = ∑_{τ=1..t} α^{t-τ} ⊙ z_τ        (channel-wise α ∈ (0,1)^H, learned)
      h_t = MLP(S_t)
      logits_t = W_tok h_t
      flow_t   = W_flow h_t

    • α is learned per channel (sigmoid-squashed, lightly clamped away from 0/1).
    • We implement the *pure* triangular/power form (no scan) for clarity.
    • PAD positions are zeroed before prefix accumulation.
    """
    def __init__(
        self,
        vocab_size: int,
        lstm_hidden: int,
        mlp_layers: int,
        mlp_width: int,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid = lstm_hidden
        self.pad_id = int(pad_id)

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, lstm_hidden)

        # Learned per-channel decay parameter (squashed to (0,1))
        # We keep it unconstrained and squash in forward with a small ε-margin.
        self._alpha_param = nn.Parameter(torch.zeros(lstm_hidden))

        # Shared MLP applied time-step-wise on decayed-prefix states
        layers = [nn.Linear(lstm_hidden, mlp_width), nn.ReLU()]
        for _ in range(max(0, mlp_layers - 1)):
            layers += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.shared_mlp = nn.Sequential(*layers)

        # Heads
        self.head_tok  = nn.Linear(mlp_width, vocab_size)  # logits for next token
        self.head_flow = nn.Linear(mlp_width, 1)           # scalar flow per position

    def _decayed_prefix(self, z: torch.Tensor, nonpad_mask: torch.Tensor) -> torch.Tensor:
        """
        Pure matrix form decayed prefix:
          z : (B, T, H)
          nonpad_mask : (B, T) in {0,1}

        Returns:
          S : (B, T, H) where S_t = ∑_{τ≤t} (α^(t-τ)) ⊙ z_τ
        """
        B, T, H = z.shape
        device = z.device
        dtype = z.dtype

        # mask PADs before accumulation
        z = z * nonpad_mask.unsqueeze(-1).to(dtype)  # (B, T, H)

        # α ∈ (0,1) per channel (safe margins to avoid exact 0/1 for stability)
        eps = 1e-4
        alpha = torch.sigmoid(self._alpha_param)          # (H,)
        alpha = eps + (1.0 - 2 * eps) * alpha             # clamp into (eps, 1-eps)

        # Build lower-triangular power matrix (T×T) of integer lags Δ=t-τ, then broadcast to H
        ar = torch.arange(T, device=device)
        delta = (ar.view(T, 1) - ar.view(1, T)).clamp_min(0)  # (T, T) with 0 on/above diag masked later
        tri = torch.tril(torch.ones((T, T), device=device, dtype=dtype))  # (T, T)

        # Per-channel powers: (T, T, H) = (alpha^delta) * tril
        # Use exp(delta * log(alpha)) for stable grads
        loga = torch.log(alpha.clamp_min(1e-12))                 # (H,)
        D = torch.exp(delta.unsqueeze(-1).to(dtype) * loga.view(1, 1, H))  # (T, T, H)
        D = D * tri.unsqueeze(-1)  # zero out above-diagonal

        # Decayed prefix via einsum over time: (T, T, H) × (B, T, H) -> (B, T, H)
        # Index legend: t(u)h, b(u)h -> b(t)h
        S = torch.einsum('tuh,buh->bth', D, z)
        return S

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq: LongTensor (B, T)
        returns:
          logits: (B, T, V)
          flow  : (B, T)
        """
        emb = self.embedding(seq)                               # (B, T, H)
        nonpad_mask = (seq != self.pad_id)                      # (B, T)
        S = self._decayed_prefix(emb, nonpad_mask)              # (B, T, H)
        h = self.shared_mlp(S)                                  # (B, T, W)
        logits = self.head_tok(h)                               # (B, T, V)
        flow   = self.head_flow(h).squeeze(-1)                  # (B, T)
        return logits, flow

    @torch.jit.export
    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced log-probabilities for actual next tokens.
        Returns (B, T-1), with PAD-masking and the EOS-prediction step removed.
        """
        B, T = seq.size(0), seq.size(1)
        if T < 2:
            return torch.empty(B, 0, device=seq.device, dtype=torch.float32)

        logits, _ = self.forward(seq[:, :-1])                   # (B, T-1, V)
        logp = torch.log_softmax(logits, dim=-1)                # (B, T-1, V)
        next_ids = seq[:, 1:]                                   # (B, T-1)
        gathered = logp.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # PAD mask: ignore where the *next* token is PAD
        pad_mask = (next_ids != self.pad_id)

        # EOS-step removal: drop time t where the model would predict EOS
        with torch.no_grad():
            nonpad = (seq != self.pad_id).to(torch.int32)       # (B, T)
            lengths = nonpad.sum(dim=1)                         # (B,)
            eos_step = lengths - 2                              # index in [0..T-2] when valid
            eos_mask = torch.ones((B, T - 1), dtype=torch.bool, device=seq.device)
            if B > 0:
                ar = torch.arange(B, device=seq.device)
                valid = (eos_step >= 0) & (eos_step < (T - 1))
                if bool(valid.any()):
                    eos_mask[ar[valid], eos_step[valid]] = False

        mask = pad_mask & eos_mask
        return gathered * mask.to(gathered.dtype)

    @torch.jit.export
    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Flow per position. Mask inside your loss if desired.
        """
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
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = int(pad_id)

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len,   d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # (B, T, D)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.head_tok  = nn.Linear(d_model, vocab_size)
        self.head_flow = nn.Linear(d_model, 1)

    def _positional(self, T: int, device: torch.device) -> torch.Tensor:
        T = T if T < self.max_len else self.max_len
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
        Teacher-forced log-probabilities, masked at PAD and with EOS step removed.
        """
        B, T = seq.size(0), seq.size(1)
        if T < 2:
            return torch.empty(B, 0, device=seq.device, dtype=torch.float32)

        logits, _ = self.forward(seq[:, :-1])         # (B, T-1, V)
        logp = torch.log_softmax(logits, dim=-1)      # (B, T-1, V)
        next_ids = seq[:, 1:]                         # (B, T-1)
        gathered = logp.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

        # PAD mask
        pad_mask = (next_ids != self.pad_id)

        # EOS-step removal
        with torch.no_grad():
            nonpad = (seq != self.pad_id).to(torch.int32)
            lengths = nonpad.sum(dim=1)
            eos_step = lengths - 2
            eos_mask = torch.ones((B, T - 1), dtype=torch.bool, device=seq.device)
            if B > 0:
                ar = torch.arange(B, device=seq.device)
                valid = (eos_step >= 0) & (eos_step < (T - 1))
                if bool(valid.any()):
                    eos_mask[ar[valid], eos_step[valid]] = False

        mask = pad_mask & eos_mask
        return gathered * mask.to(gathered.dtype)

    @torch.jit.export
    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        _, flow = self.forward(seq)
        return flow
