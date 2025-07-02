"""
Simple tokenizer for DT-GFN: dictionary-based, no external deps.

Vocab layout:
  <pad>, <bos>, <eos>, F_0…F_{num_features-1}, TH_0…TH_{num_bins-1}, LEAF

Methods mirror a typical HF tokenizer: encode_* → IDs, decode_one/decode → (type, index).
"""
from __future__ import annotations
from typing import List, Tuple


class Tokenizer:
    """
    Tokenizer with a pure-Python dict, minimal dependencies.

    Attributes:
        PAD, BOS, EOS: special token IDs
        id2tok: List[str] mapping ID → token
        tok2id: Dict[str, int] mapping token → ID
        num_features: number of feature tokens
        num_bins: number of shared threshold tokens
        num_leaves: number of leaf tokens
    """

    def __init__(
        self,
        num_features: int,
        num_bins: int,
        num_leaves: int = 1,
    ):
        # store counts
        self.num_features = num_features
        self.num_bins = num_bins
        self.num_leaves = num_leaves

        # special tokens
        specials = ["<pad>", "<bos>", "<eos>"]
        self.PAD, self.BOS, self.EOS = 0, 1, 2

        # build token lists
        feature_tokens = [f"F_{i}" for i in range(num_features)]
        thresh_tokens = [f"TH_{j}" for j in range(num_bins)]
        leaf_tokens = (
            ["LEAF"] if num_leaves == 1 else [f"LEAF_{k}" for k in range(num_leaves)]
        )

        # full vocabulary
        self.id2tok = specials + feature_tokens + thresh_tokens + leaf_tokens
        self.tok2id = {tok: idx for idx, tok in enumerate(self.id2tok)}

    def encode_feature(self, idx: int) -> int:
        """Encode a feature split token F_idx."""
        self._check(idx, self.num_features, "feature index")
        return self.tok2id[f"F_{idx}"]

    def encode_threshold(self, idx: int) -> int:
        """Encode a shared threshold bin token TH_idx."""
        self._check(idx, self.num_bins, "threshold index")
        return self.tok2id[f"TH_{idx}"]

    def encode_leaf(self, idx: int = 0) -> int:
        """Encode a leaf marker token."""
        self._check(idx, self.num_leaves, "leaf index")
        key = "LEAF" if self.num_leaves == 1 else f"LEAF_{idx}"
        return self.tok2id[key]

    def decode_one(self, tid: int) -> Tuple[str, int]:
        """Convert a non-special token ID to its (type, index)."""
        if tid <= self.EOS or tid >= len(self.id2tok):
            raise ValueError(f"Invalid token ID: {tid}")
        tok = self.id2tok[tid]
        if tok.startswith("F_"):
            return "feature", int(tok.split("_")[1])
        if tok.startswith("TH_"):
            return "threshold", int(tok.split("_")[1])
        if tok.startswith("LEAF"):
            # for both LEAF and LEAF_k
            parts = tok.split("_")
            idx = int(parts[1]) if len(parts) > 1 else 0
            return "leaf", idx
        # should not reach here
        raise ValueError(f"Unknown token string: {tok}")

    def decode(self, ids: List[int]) -> List[Tuple[str, int]]:
        """Decode a sequence of IDs, skipping PAD/BOS/EOS."""
        out: List[Tuple[str,int]] = []
        for tid in ids:
            if tid in (self.PAD, self.BOS, self.EOS):
                continue
            out.append(self.decode_one(tid))
        return out

    def _check(self, idx: int, size: int, name: str) -> None:
        """Ensure 0 <= idx < size."""
        if idx < 0 or idx >= size:
            raise IndexError(f"{name} {idx} out of range [0, {size}).")

    def __len__(self) -> int:
        """Vocabulary size."""
        return len(self.id2tok)
