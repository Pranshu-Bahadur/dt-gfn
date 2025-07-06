"""tokenizer.py – Minimal, fast, and extensible tokenizer for DT‑GFN
-------------------------------------------------------------------------
Token layout
  0  : <pad>
  1  : <bos>
  2  : <eos>
  3‑(3+F‑1)           : feature IDs        (F = num_features)
  …                   : threshold IDs      (T = num_bins)
  …                   : leaf IDs           (L = num_leaves)
-------------------------------------------------------------------------
The class is deliberately *stateless* apart from vocabulary sizes so it can
be shared across workers / processes without pickling overhead.
The design allows a future drop‑in BPE or SentencePiece tokenizer by only
changing the encode/decode helpers while keeping token IDs stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

__all__ = ["Tokenizer", "Vocab"]


@dataclass(frozen=True, slots=True)
class Vocab:
    """Holds the offset of each token block."""

    num_features: int
    num_bins: int
    num_leaves: int

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    @property
    def feature_start(self) -> int:  # 3
        return 3

    @property
    def threshold_start(self) -> int:
        return self.feature_start + self.num_features

    @property
    def leaf_start(self) -> int:
        return self.threshold_start + self.num_bins

    @property
    def size(self) -> int:
        return self.leaf_start + self.num_leaves


class Tokenizer:
    """Fast int‑token encoder/decoder with O(1) conversions.

    >>> tok = Tokenizer(num_features=5, num_bins=7, num_leaves=2)
    >>> ids = [
    ...     tok.BOS,
    ...     tok.encode_feature(3),
    ...     tok.encode_threshold(6),
    ...     tok.encode_leaf(1),
    ...     tok.EOS,
    ... ]
    >>> tok.decode(ids)
    [('feature', 3), ('threshold', 6), ('leaf', 1)]
    """

    __slots__ = ("v",)

    def __init__(self, *, num_features: int, num_bins: int, num_leaves: int = 1):
        self.v = Vocab(
            num_features=int(num_features),
            num_bins=int(num_bins),
            num_leaves=int(num_leaves),
        )

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    # shorthand aliases
    @property
    def PAD(self) -> int:  # noqa: N802
        return self.v.PAD

    @property
    def BOS(self) -> int:  # noqa: N802
        return self.v.BOS

    @property
    def EOS(self) -> int:  # noqa: N802
        return self.v.EOS

    # encode helpers ---------------------------------------------------- #
    def encode_feature(self, idx: int) -> int:
        self._check_bounds(idx, self.v.num_features, "feature")
        return self.v.feature_start + idx

    def encode_threshold(self, idx: int) -> int:
        self._check_bounds(idx, self.v.num_bins, "threshold")
        return self.v.threshold_start + idx

    def encode_leaf(self, idx: int) -> int:
        self._check_bounds(idx, self.v.num_leaves, "leaf")
        return self.v.leaf_start + idx

    # decode helpers ---------------------------------------------------- #
    @lru_cache(maxsize=4096)
    def decode_one(self, tid: int) -> Tuple[str, int]:
        """Inverse of the *encode_* helpers.

        Returns
        -------
        (kind, index) where kind ∈ {"feature", "threshold", "leaf"}
        """
        if tid < 0 or tid >= len(self):
            raise ValueError(f"Invalid token id: {tid}")

        if tid in (self.PAD, self.BOS, self.EOS):
            raise ValueError("Special tokens do not decode to (kind, index).")

        if tid < self.v.threshold_start:
            return "feature", tid - self.v.feature_start
        if tid < self.v.leaf_start:
            return "threshold", tid - self.v.threshold_start
        return "leaf", tid - self.v.leaf_start

    def decode(self, ids: List[int]) -> List[Tuple[str, int]]:
        """Decodes a *sequence* of token IDs, skipping special tokens."""
        return [
            self.decode_one(i)
            for i in ids
            if i not in (self.PAD, self.BOS, self.EOS)
        ]

    # ------------------------------------------------------------------ #
    # Dunder methods
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:  # `len(tok)` == vocab size
        return self.v.size

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "Tokenizer(vocab_size={v}, features={f}, bins={b}, leaves={l})".format(
                v=len(self),
                f=self.v.num_features,
                b=self.v.num_bins,
                l=self.v.num_leaves,
            )
        )

    # ------------------------------------------------------------------ #
    # Private utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_bounds(idx: int, upper: int, label: str) -> None:
        if not (0 <= idx < upper):
            raise ValueError(f"{label} index {idx} out of range [0, {upper})")

