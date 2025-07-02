from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class TokenType(str, Enum):
    """Kinds of tokens in a decision tree sequence."""
    FEATURE   = "feature"
    THRESHOLD = "threshold"
    LEAF      = "leaf"


@dataclass(frozen=True)
class Vocab:
    """
    Holds counts for each token category and special tokens.

    Attributes:
        num_features: Number of feature split tokens
        num_thresholds: Number of threshold tokens
        num_leaves: Number of leaf tokens
    """
    num_features:   int
    num_thresholds: int
    num_leaves:     int

    # Special token indices
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    @property
    def split_start(self) -> int:
        """Index where split/token IDs begin."""
        return self.EOS + 1

    def size(self) -> int:
        """Total vocabulary size (including PAD/BOS/EOS)."""
        return (
            self.split_start
            + self.num_features
            + self.num_thresholds
            + self.num_leaves
        )

    def __len__(self) -> int:
        return self.size()


class Tokenizer:
    """
    Converts between token IDs and (type, index) pairs.
    """

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    # --- Encoding methods ------------------------------------------------

    def encode_feature(self, idx: int) -> int:
        """Get token ID for the idx-th feature split."""
        self._check_range(idx, self.vocab.num_features, "feature index")
        return self.vocab.split_start + idx

    def encode_threshold(self, idx: int) -> int:
        """Get token ID for the idx-th threshold value."""
        offset = self.vocab.num_features
        self._check_range(idx, self.vocab.num_thresholds, "threshold index")
        return self.vocab.split_start + offset + idx

    def encode_leaf(self, idx: int) -> int:
        """Get token ID for the idx-th leaf marker."""
        offset = self.vocab.num_features + self.vocab.num_thresholds
        self._check_range(idx, self.vocab.num_leaves, "leaf index")
        return self.vocab.split_start + offset + idx

    # --- Decoding methods ------------------------------------------------

    def decode_one(self, tid: int) -> Tuple[TokenType, int]:
        """Convert a non-special token ID to its type and index."""
        start = self.vocab.split_start
        end = self.vocab.size()
        if tid < start or tid >= end:
            raise ValueError(
                f"Token ID {tid} out of valid range [{start}, {end})."
            )
        rem = tid - start
        if rem < self.vocab.num_features:
            return TokenType.FEATURE, rem
        rem -= self.vocab.num_features
        if rem < self.vocab.num_thresholds:
            return TokenType.THRESHOLD, rem
        return TokenType.LEAF, rem - self.vocab.num_thresholds

    def decode(self, ids: List[int]) -> List[Tuple[TokenType, int]]:
        """
        Decode a list of token IDs, skipping PAD/BOS/EOS.
        """
        result: List[Tuple[TokenType, int]] = []
        special = {self.vocab.PAD, self.vocab.BOS, self.vocab.EOS}
        for tid in ids:
            if tid in special:
                continue
            result.append(self.decode_one(tid))
        return result

    # --- Helpers ---------------------------------------------------------

    def _check_range(self, idx: int, size: int, name: str) -> None:
        """Validate an index is within [0, size)."""
        if idx < 0 or idx >= size:
            raise IndexError(
                f"{name} {idx} out of range [0, {size})."
            )

