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
        max_len: int = 384,
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
        

# -------------------------------
# Recurrent Retention Policy (RetNet-style)
# -------------------------------
from dataclasses import dataclass

@dataclass
class RetentionConfig:
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4          # d_model must be divisible by n_heads
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 1024
    pad_id: int = 0
    learnable_gamma: bool = True  # per-head decay is learnable (0..1)
    gamma_init_power: float = -5.0  # gamma_h = 1 - 2^(gamma_init_power - h)

class _MultiHeadRecurrentRetention(nn.Module):
    """
    Multi-head recurrent retention core.

    For each time step t:
      S_h <- gamma_h * S_h + k_{t,h} ⊗ v_{t,h}         (state per head h: dh×dh)
      y_{t,h} = q_{t,h} @ S_h
    Concatenate heads -> (B, T, D).

    Notes:
      • We mask PADs by zeroing q,k,v at those positions and skipping their contribution.
      • Complexity per step per head is O(dh^2). With d_model=256, n_heads=4 → dh=64 (fine).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 learnable_gamma: bool, gamma_init_power: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        D, H, dh = d_model, n_heads, d_model // n_heads
        # projections (no bias; bias can be added if desired)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)

        # per-head decay gamma ∈ (0,1); use sigmoid parameterization
        # init schedule similar to TF example: gamma_h = 1 - 2^(-5 - h)
        init = []
        for h in range(H):
            g = 1.0 - (2.0 ** (gamma_init_power - float(h)))
            g = max(1e-4, min(1.0 - 1e-4, g))
            # inverse-sigmoid
            init.append(-torch.log(torch.tensor(1.0 / g - 1.0)))
        self._gamma_param = nn.Parameter(torch.stack(init)) if learnable_gamma else None
        self.register_buffer("_gamma_fixed", torch.stack(init).sigmoid() if not learnable_gamma else torch.zeros(H))

        self.dropout = nn.Dropout(dropout)
        self.gn = nn.GroupNorm(num_groups=n_heads, num_channels=D, affine=True)
        self.wo = nn.Linear(D, D, bias=False)
        self.wg = nn.Sequential(nn.Linear(D, D, bias=False), nn.SiLU())

        self.ln = nn.LayerNorm(D)

    def _gammas(self) -> torch.Tensor:
        if self._gamma_param is not None:
            g = torch.sigmoid(self._gamma_param)  # (H,)
        else:
            g = self._gamma_fixed
        # keep a tiny safety margin
        eps = 1e-4
        return eps + (1.0 - 2*eps) * g  # (H,)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        key_padding_mask: (B, T) True at PAD positions
        returns: (B, T, D)
        """
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        device = x.device

        Q = self.q_proj(x).view(B, T, H, dh)  # (B,T,H,dh)
        K = self.k_proj(x).view(B, T, H, dh)
        V = self.v_proj(x).view(B, T, H, dh)

        # state S per head: (B,H,dh,dh)
        S = x.new_zeros(B, H, dh, dh)
        gam = self._gammas().to(device).view(1, H, 1, 1)  # (1,H,1,1)

        Y = x.new_zeros(B, T, H, dh)
        # iterate over time (TorchScript-friendly; no Python lists)
        for t in range(T):
            nonpad = (~key_padding_mask[:, t]).float().view(B, 1, 1)  # (B,1,1)
            q_t = Q[:, t] * nonpad        # (B,H,dh)
            k_t = K[:, t] * nonpad
            v_t = V[:, t] * nonpad

            # outer product per head
            # (B,H,dh,dh) += (B,H,dh) ⊗ (B,H,dh)
            S = gam * S + torch.einsum('bhd,bhe->bhde', k_t, v_t)

            # y_t = q_t @ S
            Y[:, t] = torch.einsum('bhd,bhde->bhe', q_t, S)

        # concat heads, group-norm, gated output, residual
        Y = Y.reshape(B, T, D)
        Y_flat = Y.reshape(B*T, D)
        Y_norm = self.gn(Y_flat).reshape(B, T, D)
        out = self.wo(self.wg(x) * Y_norm)
        out = self.dropout(out)
        return self.ln(x + out)

class _RetentionFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ff(self.ln(x))
        return x + h

class _RetentionBlock(nn.Module):
    def __init__(self, cfg: RetentionConfig):
        super().__init__()
        self.core = _MultiHeadRecurrentRetention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
            learnable_gamma=cfg.learnable_gamma,
            gamma_init_power=cfg.gamma_init_power,
        )
        self.ffn = _RetentionFFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, kpm: torch.Tensor) -> torch.Tensor:
        x = self.core(x, kpm)
        x = self.ffn(x)
        return x

class PolicyRetention(PolicyBase):
    """
    Recurrent Retention Policy Network (RetNet-style), drop-in for DT-GFN.

    Token + positional embeddings → L×[RetentionBlock] → heads:
      – next-token logits  (B × T × V)
      – flow estimate      (B × T)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = int(pad_id)
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        cfg = RetentionConfig(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pad_id=pad_id,
        )
        self.layers = nn.ModuleList([_RetentionBlock(cfg) for _ in range(n_layers)])

        self.head_tok  = nn.Linear(d_model, vocab_size)
        self.head_flow = nn.Linear(d_model, 1)

    def _positional(self, T: int, device: torch.device) -> torch.Tensor:
        T = min(T, self.max_len)
        pos = torch.arange(T, device=device)
        return self.pos_emb(pos).unsqueeze(0)  # (1,T,D)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq : (B,T) int64 token IDs
        Returns:
          logits : (B,T,V)
          flow   : (B,T)
        """
        B, T = seq.size(0), seq.size(1)
        x = self.tok_emb(seq) + self._positional(T, seq.device)  # (B,T,D)
        kpm = (seq == self.pad_id)                                # (B,T) True where PAD

        for blk in self.layers:
            x = blk(x, kpm)

        logits = self.head_tok(x)                 # (B,T,V)
        flow   = self.head_flow(x).squeeze(-1)    # (B,T)
        return logits, flow

    @torch.jit.export
    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced log-probs for next tokens (B, T-1),
        masked at PAD, with the single EOS-prediction step removed.
        """
        B, T = seq.size(0), seq.size(1)
        if T < 2:
            return torch.empty(B, 0, device=seq.device, dtype=torch.float32)

        logits, _ = self.forward(seq[:, :-1])         # (B, T-1, V)
        logp = torch.log_softmax(logits, dim=-1)      # (B, T-1, V)
        next_ids = seq[:, 1:]                         # (B, T-1)
        gathered = logp.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)

        # PAD mask on the *next* token
        pad_mask = (next_ids != self.pad_id)

        # Remove the EOS-prediction step (sampler appends EOS deterministically)
        with torch.no_grad():
            nonpad = (seq != self.pad_id).to(torch.int32)
            lengths = nonpad.sum(dim=1)               # includes BOS & EOS
            eos_step = lengths - 2                    # index in [0..T-2]
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