"""tree.py – Core tree data models for DT‑GFN
=========================================================
This single module contains *all* immutable, data‑only objects that
represent decision trees at three granularities:

* **DecisionRule** – atomic split (`feature ≤ threshold_id`).
* **PartialTree**  – one *root→leaf* path (state in the GFlowNet MDP).
* **Tree**         – a *set* of mutually‑exclusive PartialTrees (i.e. the
  full decision tree seen by downstream inference / ensembling).

They are intentionally lightweight: no training logic, no global
singletons, and zero dependencies on Numerai‑specific code.  This makes
unit‑testing trivial and keeps the objects reusable in other projects.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# 1. Atomic split
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionRule:
    """Binary feature split represented *after binning*.

    Parameters
    ----------
    feature : int
        Column index in the *binned* feature matrix ``X_bin``.
    threshold_id : int
        Integer threshold in the same bin space (0 ≤ thr < n_bins).
    """

    feature: int
    threshold_id: int

    # Fast predicate --------------------------------------------------------
    def apply(self, X_bin: torch.LongTensor) -> torch.BoolTensor:  # shape: [N, F]
        """Return a Bool mask of rows that satisfy the rule."""

        return X_bin[:, self.feature] <= self.threshold_id


# ---------------------------------------------------------------------------
# 2. Single path (MDP state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PartialTree:
    """Immutable root→leaf path used as a *state* inside DT‑GFN."""

    rules: Tuple[DecisionRule, ...]
    depth_limit: int = 6

    # --- Convenience properties -------------------------------------------
    def n_nodes(self) -> int:
        return len(self.rules)

    def is_terminal(self) -> bool:
        return self.n_nodes() >= self.depth_limit

    # --- Tokenisation ------------------------------------------------------
    def to_tokens(self, tokenizer: "Tokenizer", *, add_special: bool = True) -> torch.LongTensor:  # noqa: F821 – Tokenizer is imported at runtime
        """Encode the path into a sequence of token IDs ready for the policy.

        Special tokens ``<bos>`` and ``<eos>`` are optionally added so the
        returned tensor can be fed directly to an autoregressive model.
        """

        ids = []
        if add_special:
            ids.append(tokenizer.BOS)
        for r in self.rules:
            ids.append(tokenizer.encode_feature(r.feature))
            ids.append(tokenizer.encode_threshold(r.threshold_id))
        if add_special:
            ids.append(tokenizer.EOS)
        return torch.as_tensor(ids, dtype=torch.long)

    # --- Vectorised mask ---------------------------------------------------
    @lru_cache(maxsize=1024)
    def _mask_fn(self) -> callable[[torch.LongTensor], torch.BoolTensor]:
        """Return a *cached* function that applies all rules in order."""

        def _apply(X_bin: torch.LongTensor) -> torch.BoolTensor:
            mask = torch.ones(X_bin.size(0), dtype=torch.bool, device=X_bin.device)
            for rule in self.rules:
                mask &= rule.apply(X_bin)
            return mask

        return _apply

    def apply(self, X_bin: torch.LongTensor) -> torch.BoolTensor:
        """Apply all rules to a batch of *binned* features.

        Parameters
        ----------
        X_bin : torch.LongTensor, shape = [N, F]
            Pre‑binned feature matrix.

        Returns
        -------
        torch.BoolTensor, shape = [N]
            Mask of rows that reach this leaf.
        """

        return self._mask_fn()(X_bin)


# ---------------------------------------------------------------------------
# 3. Full tree (collection of leaves)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tree:
    """A *complete* decision tree assembled from PartialTrees.

    Notes
    -----
    * Immutability allows safe sharing across processes & replay buffers.
    * ``weight`` can store importance in an ensemble; default 1.0.
    """

    leaves: Tuple[PartialTree, ...]
    weight: float = 1.0

    # --- Inference ---------------------------------------------------------
    def predict_bin(self, X_bin: torch.LongTensor) -> torch.Tensor:
        """Vectorised prediction in the *binned* space.

        Each leaf votes +1 for rows that fall into it; votes are summed and
        multiplied by ``weight``.  Downstream neutralisation rescales anyway
        so we keep it simple (binary margin).
        """

        # Allocate on device once
        scores = torch.zeros(X_bin.size(0), dtype=torch.float32, device=X_bin.device)
        for leaf in self.leaves:
            mask = leaf.apply(X_bin)
            scores[mask] += 1.0
        scores *= self.weight
        return scores

    # --- Utility -----------------------------------------------------------
    def n_leaves(self) -> int:
        return len(self.leaves)

