"""tokenizer.py – Minimal, fast, and extensible tokenizer for DT‑GFN
-------------------------------------------------------------------------
Token layout
  0  : <pad>
  1  : <bos>
  2  : <eos>
  3‑(3+F‑1)           : feature IDs        (F = num_features)
  …                   : threshold IDs      (T = n_bins)
  …                   : leaf IDs           (L = num_leaves)
-------------------------------------------------------------------------
The class is deliberately *stateless* apart from vocabulary sizes so it can
be shared across workers / processes without pickling overhead.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

__all__ = ["Tokenizer", "Vocab"]

@dataclass(frozen=True, slots=True)
class Vocab:
    """Holds the offset and size of each token block."""
    num_features: int
    num_bins: int
    num_leaves: int

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    @property
    def feature_start(self) -> int:
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
    """
    Fast integer-token encoder/decoder with O(1) conversions.
    """
    __slots__ = ("v",)

    def __init__(self, *, num_features: int, num_bins: int, num_leaves: int = 1):
        self.v = Vocab(
            num_features=int(num_features),
            num_bins=int(num_bins),
            num_leaves=int(num_leaves),
        )

    @property
    def PAD(self) -> int: return self.v.PAD
    @property
    def BOS(self) -> int: return self.v.BOS
    @property
    def EOS(self) -> int: return self.v.EOS

    def encode_feature(self, idx: int) -> int:
        self._check_bounds(idx, self.v.num_features, "feature")
        return self.v.feature_start + idx

    def encode_threshold(self, idx: int) -> int:
        self._check_bounds(idx, self.v.num_bins, "threshold")
        return self.v.threshold_start + idx

    def encode_leaf(self, idx: int) -> int:
        self._check_bounds(idx, self.v.num_leaves, "leaf")
        return self.v.leaf_start + idx

    def decode_one(self, tid: int) -> Tuple[str, int]:
        if not isinstance(tid, int) or not (0 <= tid < self.v.size):
            raise ValueError(f"Invalid token ID: {tid}")

        if tid >= self.v.leaf_start:
            return "leaf", tid - self.v.leaf_start
        if tid >= self.v.threshold_start:
            return "threshold", tid - self.v.threshold_start
        if tid >= self.v.feature_start:
            return "feature", tid - self.v.feature_start
        
        raise ValueError(f"Special token ID {tid} cannot be decoded to (kind, index).")

    def decode(self, ids: List[int]) -> List[Tuple[str, int]]:
        """Decodes a sequence of token IDs, skipping special tokens."""
        decoded_list = []
        for i in ids:
            if i not in (self.v.PAD, self.v.BOS, self.v.EOS):
                decoded_list.append(self.decode_one(i))
        return decoded_list

    def __len__(self) -> int:
        return self.v.size

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={len(self)}, features={self.v.num_features}, "
            f"bins={self.v.num_bins}, leaves={self.v.num_leaves})"
        )

    @staticmethod
    def _check_bounds(idx: int, upper: int, label: str) -> None:
        if not (0 <= idx < upper):
            raise IndexError(f"{label.capitalize()} index {idx} is out of bounds for size {upper}.")