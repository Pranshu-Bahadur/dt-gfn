# src/tokenizer.py

"""
Simple, efficient tokenizer for DT-GFN, matching the production pipeline.

Vocab layout:
  <pad>, <bos>, <eos>, F_0...F_{num_features-1}, TH_0...TH_{num_bins-1}, LEAF
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Vocab:
    """
    A dataclass to hold vocabulary size information.
    """
    num_feat: int
    num_th: int
    num_leaf: int
    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    @property
    def split_start(self) -> int:
        """The starting index for feature, threshold, and leaf tokens."""
        return 3

    def size(self) -> int:
        """The total size of the vocabulary."""
        return self.split_start + self.num_feat + self.num_th + self.num_leaf

class Tokenizer:
    """
    A streamlined tokenizer that uses a Vocab object to manage token IDs.
    """
    def __init__(self, vocab: Vocab):
        self.v = vocab

    def _feat(self, i: int) -> int:
        """Gets the token ID for a feature."""
        return self.v.split_start + i

    def _th(self, i: int) -> int:
        """Gets the token ID for a threshold."""
        return self.v.split_start + self.v.num_feat + i

    def _leaf(self, i: int) -> int:
        """Gets the token ID for a leaf."""
        return self.v.split_start + self.v.num_feat + self.v.num_th + i

    def decode_one(self, tid: int) -> Tuple[str, int]:
        """Converts a token ID to its type and index."""
        if tid < self.v.split_start:
            raise ValueError(f"Invalid token id: {tid}")
        
        rem = tid - self.v.split_start
        if rem < self.v.num_feat:
            return "feat", rem
        
        rem -= self.v.num_feat
        if rem < self.v.num_th:
            return "th", rem
            
        return "leaf", rem - self.v.num_th

    def decode(self, ids: List[int]) -> List[Tuple[str, int]]:
        """Decodes a sequence of token IDs."""
        return [self.decode_one(i) for i in ids if i not in (self.v.BOS, self.v.EOS)]