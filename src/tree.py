"""tree.py – Core tree data models for DT‑GFN
=========================================================
This module contains immutable data objects representing decision trees.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from src.tokenizer import Tokenizer

@dataclass(frozen=True, slots=True)
class DecisionRule:
    """A binary split on a feature based on a binned threshold."""
    feature: int
    threshold_id: int

    def apply(self, X_bin: torch.Tensor) -> torch.Tensor:
        """Returns a boolean mask of rows satisfying the rule."""
        return X_bin[:, self.feature] <= self.threshold_id

@dataclass(frozen=True) # <-- Fix: removed slots=True
class PartialTree:
    """An immutable root-to-leaf path, representing a state in the GFN."""
    rules: Tuple[DecisionRule, ...]
    depth_limit: int = 6

    @cached_property
    def n_nodes(self) -> int:
        return len(self.rules)

    def is_terminal(self) -> bool:
        return self.n_nodes >= self.depth_limit

    def to_tokens(self, tokenizer: "Tokenizer", *, add_special: bool = True) -> torch.Tensor:
        """Encodes the path into a sequence of token IDs."""
        tokens = []
        if add_special:
            tokens.append(tokenizer.BOS)
        for rule in self.rules:
            tokens.append(tokenizer.encode_feature(rule.feature))
            tokens.append(tokenizer.encode_threshold(rule.threshold_id))
        if add_special:
            tokens.append(tokenizer.EOS)
        return torch.tensor(tokens, dtype=torch.long)

    def apply(self, X_bin: torch.Tensor) -> torch.Tensor:
        """Applies all rules to a batch of binned features."""
        if not self.rules:
            return torch.ones(X_bin.size(0), dtype=torch.bool, device=X_bin.device)

        mask = self.rules[0].apply(X_bin)
        for i in range(1, len(self.rules)):
            mask.logical_and_(self.rules[i].apply(X_bin))
        return mask

@dataclass(frozen=True) # <-- Fix: removed slots=True
class Tree:
    """A complete decision tree, assembled from a set of leaves."""
    leaves: Tuple[PartialTree, ...]
    weight: float = 1.0

    def predict_bin(self, X_bin: torch.Tensor) -> torch.Tensor:
        """Vectorized prediction on binned features."""
        scores = torch.zeros(X_bin.size(0), dtype=torch.float32, device=X_bin.device)
        for leaf in self.leaves:
            mask = leaf.apply(X_bin)
            scores[mask] += 1.0  # Simple vote
        return scores * self.weight

    @cached_property
    def n_leaves(self) -> int:
        return len(self.leaves)