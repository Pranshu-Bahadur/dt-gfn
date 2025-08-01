from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple
import math
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

    # src/trees/policy.py  (add after PolicyPaperMLP)



class PolicyTransformer(PolicyBase):
    """
    Transformer-encoder policy for DT-GFN.

    • Token + positional embeddings → stack of Transformer blocks
    • Two linear heads:
        – next-token logits  (B × T × V)
        – flow estimate      (B × T)

    Args
    ----
    vocab_size      : size of the token vocabulary
    d_model         : embedding / hidden size
    n_layers        : number of Transformer encoder layers
    n_heads         : number of attention heads
    d_ff            : feed-forward width inside each block
    dropout         : dropout rate in Transformer blocks
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 2,
        d_ff: int = 256*4,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()

        # --- embeddings --------------------------------------------------
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len,   d_model)

        # --- Transformer encoder ----------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model   = d_model,
            nhead     = n_heads,
            dim_feedforward = d_ff,
            dropout   = dropout,
            batch_first = True,            # (B, T, D)
            activation = "relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- heads -------------------------------------------------------
        self.head_tok  = nn.Linear(d_model, vocab_size)
        self.head_flow = nn.Linear(d_model, 1)

        # init
        #self._reset_parameters()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _reset_parameters(self):
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0, std=0.02)
        nn.init.xavier_uniform_(self.head_tok.weight)
        nn.init.zeros_(self.head_tok.bias)
        nn.init.xavier_uniform_(self.head_flow.weight)
        nn.init.zeros_(self.head_flow.bias)

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, D)
        B, T, _ = x.size()
        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self.pos_emb(pos_ids)      # (T, D)
        return x + pos_emb.unsqueeze(0)      # broadcast over batch

    # ------------------------------------------------------------------ #
    # main API
    # ------------------------------------------------------------------ #
    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq : (B, T) int64 token IDs
        Returns (logits, flow):
          logits : (B, T, V)
          flow   : (B, T)
        """
        h = self.token_emb(seq)              # (B, T, D)
        #h = self._add_positional(h)
        h = self.encoder(h)                  # Transformer blocks
        logits = self.head_tok(h)            # (B, T, V)
        flow   = self.head_flow(h).squeeze(-1)  # (B, T)
        return logits, flow

    @torch.jit.export
    def log_prob(self, seq: torch.Tensor) -> torch.Tensor:
        # identical to LSTM version, just reuse forward()
        if seq.size(1) < 2:
            return torch.empty(seq.size(0), 0, device=seq.device)
        logits, _ = self.forward(seq[:, :-1])    # (B, T-1, V)
        logp = logits.log_softmax(dim=-1)
        return torch.gather(
            logp, -1, seq[:, 1:].unsqueeze(-1)
        ).squeeze(-1)                            # (B, T-1)

    @torch.jit.export
    def log_F(self, seq: torch.Tensor) -> torch.Tensor:
        _, flow = self.forward(seq)
        return flow