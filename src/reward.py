"""reward.py – Vectorised Gaussian‑NIG reward for DT‑GFN (Numerai regression)
==============================================================================
This module converts the closed‑form marginal likelihood of a Normal
likelihood with a Normal‑Inverse‑Gamma (NIG) prior into *log‑reward
values* that the GFlowNet can optimise via trajectory balance.

Notation follows Bishop PRML §2.3 and the DT‑GFN paper, adapted from
classification to regression targets.

Formulas
--------
Given a leaf with statistics::

    n   = number of samples in leaf
    S_y = ∑ y
    S2  = ∑ y^2

and prior hyper‑parameters ``(mu0, kappa0, nu0, sig0_sq)``, the marginal
log‑likelihood is:

.. math::

    \log p(Y_\text{leaf}) = \tfrac12\left[\log(\kappa_0/\kappa_n)\right]
        + \operatorname{lgamma}(\tfrac{\nu_n}{2}) - \operatorname{lgamma}(\tfrac{\nu_0}{2})
        + \tfrac{\nu_0}{2}\log(\sigma_0^2)
        - \tfrac{\nu_n}{2} \log(\sigma_n^2)

with

::

    kappa_n = kappa0 + n
    nu_n    = nu0    + n
    mu_n    = (kappa0*mu0 + S_y) / kappa_n
    sigma_n^2 = nu0*sigma0_sq + S2 + kappa0*mu0**2 - kappa_n*mu_n**2

The *tree* reward is the sum over its leaves minus ``beta * n_nodes``.

API
---
``RewardPrecomputer`` caches the sufficient statistics ``(N, S1, S2)``
for **all** feature‑bin combinations so we can evaluate any candidate
split in :math:`\mathcal O(1)`.

::

    pre = RewardPrecomputer(X_bin, y)
    cfg = RewardConfig(beta=2.3)
    reward = tree_log_reward(tree, pre, cfg)

The functions work on both ``Tree`` and ``PartialTree`` instances.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from tree import DecisionRule, PartialTree, Tree  # type: ignore  # circular‑imports guarded

__all__ = [
    "RewardConfig",
    "RewardPrecomputer",
    "leaf_log_marginal",
    "partial_tree_log_reward",
    "tree_log_reward",
]


# ---------------------------------------------------------------------------
# 1.  Config dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class RewardConfig:
    mu0: float = 0.0
    kappa0: float = 1.0
    nu0: float = 1.0
    sigma0_sq: float = 1.0
    beta: float = 2.0  # size penalty per node


# ---------------------------------------------------------------------------
# 2.  Sufficient‑stat pre‑computation (once per epoch)
# ---------------------------------------------------------------------------
class RewardPrecomputer:
    """Caches ∑y, ∑y², and n for each (feature, bin).

    Args
    ----
    X_bin : ``[N, F]`` tensor of **integers** (output of ``Binner``)
    y     : ``[N]`` tensor of targets (float32)
    n_bins: maximum number of bins (must ≥ actual)
    """

    def __init__(self, X_bin: torch.LongTensor, y: torch.Tensor, n_bins: int):
        assert X_bin.dim() == 2, "X_bin must be 2‑D [N,F]"
        N, F = X_bin.shape
        device = X_bin.device
        y = y.to(device)

        # Initialise tensors
        self.counts = torch.zeros((F, n_bins), dtype=torch.int32, device=device)
        self.sum_y = torch.zeros((F, n_bins), dtype=torch.float32, device=device)
        self.sum_y2 = torch.zeros((F, n_bins), dtype=torch.float32, device=device)

        # Vectorised scatter add per feature
        for f in range(F):
            bins = X_bin[:, f]
            self.counts[f].scatter_add_(0, bins, torch.ones_like(bins, dtype=torch.int32))
            self.sum_y[f].scatter_add_(0, bins, y)
            self.sum_y2[f].scatter_add_(0, bins, y * y)

        self.F = F
        self.n_bins = n_bins

    # -------------------------------------------------------------
    def leaf_stats(self, rules: Tuple[DecisionRule, ...]) -> Tuple[int, float, float]:
        """Returns (n, S1, S2) for the **intersection** of rule masks.

        Intersection trick: we start from *all* rows and iteratively mask
        by counts subtraction.  Because full per‑row masks are expensive
        we take the *min* over counts implied by each rule.
        """
        # Start with all rows indices
        n_total = None
        sum_y = None
        sum_y2 = None

        for rule in rules:
            f, b = rule.feature, rule.threshold_id
            c = self.counts[f, b]
            s1 = self.sum_y[f, b]
            s2 = self.sum_y2[f, b]
            if n_total is None:
                n_total, sum_y, sum_y2 = c, s1, s2
            else:
                # Conservative intersection via min – works because bins are mutually exclusive per feature
                n_total = torch.minimum(n_total, c)
                sum_y = torch.minimum(sum_y, s1)
                sum_y2 = torch.minimum(sum_y2, s2)
        return int(n_total), float(sum_y), float(sum_y2)


# ---------------------------------------------------------------------------
# 3.  Leaf marginal likelihood (Gaussian‑NIG)
# ---------------------------------------------------------------------------

def leaf_log_marginal(n: int, S1: float, S2: float, cfg: RewardConfig) -> float:
    if n == 0:
        return 0.0
    mu0, kappa0, nu0, sig0_sq = cfg.mu0, cfg.kappa0, cfg.nu0, cfg.sigma0_sq
    kappa_n = kappa0 + n
    nu_n = nu0 + n
    mu_n = (kappa0 * mu0 + S1) / kappa_n
    sig_n = (
        nu0 * sig0_sq
        + S2
        + kappa0 * mu0 * mu0
        - kappa_n * mu_n * mu_n
    )
    return (
        0.5 * torch.log(torch.tensor(kappa0 / kappa_n))
        + torch.lgamma(torch.tensor(nu_n / 2.0))
        - torch.lgamma(torch.tensor(nu0 / 2.0))
        + (nu0 / 2.0) * torch.log(torch.tensor(sig0_sq))
        - (nu_n / 2.0) * torch.log(sig_n)
    ).item()


# ---------------------------------------------------------------------------
# 4.  Tree / PartialTree reward helpers
# ---------------------------------------------------------------------------

def partial_tree_log_reward(pt: PartialTree, pre: RewardPrecomputer, cfg: RewardConfig) -> float:
    n, s1, s2 = pre.leaf_stats(pt.rules)
    return leaf_log_marginal(n, s1, s2, cfg) - cfg.beta * pt.n_nodes


def tree_log_reward(tree: Tree, pre: RewardPrecomputer, cfg: RewardConfig) -> float:
    total = 0.0
    for leaf in tree.leaves:
        n, s1, s2 = pre.leaf_stats(leaf.rules)
        total += leaf_log_marginal(n, s1, s2, cfg)
    total -= cfg.beta * tree.n_nodes
    return total

