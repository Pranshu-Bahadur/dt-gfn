"""reward.py – Vectorised Gaussian-NIG reward for DT-GFN (Numerai regression)
==============================================================================
This module implements the closed-form marginal likelihood of a Normal
likelihood with a Normal-Inverse-Gamma (NIG) prior. This serves as the
log-reward for the GFlowNet.

The implementation is designed for correctness and efficiency, providing both
single-leaf and batched computation for sufficient statistics.

Key Components:
- RewardConfig: Dataclass for prior hyperparameters and complexity penalty.
- batch_leaf_stats: Vectorized function to compute stats for many leaves at once.
- RewardPrecomputer: Caches initial statistics (no longer used for exact
  leaf calculations but kept for potential future hybrid approaches).
- leaf_log_marginal: Calculates the core Bayesian reward for a single leaf.
- tree_log_reward: Calculates the total reward for a complete tree.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch

# Make sure to import the correct tree structures from your project
from src.tree import DecisionRule, PartialTree, Tree

__all__ = [
    "RewardConfig",
    "batch_leaf_stats",
    "leaf_log_marginal",
    "tree_log_reward",
    "partial_tree_log_reward",
]


@dataclass(slots=True, frozen=True)
class RewardConfig:
    """Hyperparameters for the Normal-Inverse-Gamma prior and tree complexity."""
    mu0: float = 0.5       # Prior mean (for Numerai, centered at 0.5)
    kappa0: float = 1.0    # Confidence in prior mean
    nu0: float = 1.0       # Degrees of freedom for prior variance
    sigma0_sq: float = 0.1 # Prior variance
    beta: float = 2.0      # Penalty per unique node for tree complexity


def batch_leaf_stats(
    leaves: List[PartialTree],
    X_bin_full: torch.Tensor,
    y_full: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates exact sufficient statistics for a BATCH of leaves in a fully
    vectorized manner.

    Args:
        leaves: A list of PartialTree objects.
        X_bin_full: The complete binned training feature tensor [N, F].
        y_full: The complete training target tensor [N].

    Returns:
        A tensor of shape [B, 3] where each row contains (n, sum_y, sum_y_sq)
        for the corresponding leaf in the batch.
    """
    if not leaves:
        return torch.empty((0, 3), dtype=torch.float32, device=y_full.device)

    # 1. Create a mask for each leaf in the batch.
    # The result is a boolean tensor of shape [BatchSize, NumDataPoints].
    batch_masks = torch.stack([leaf.apply(X_bin_full) for leaf in leaves])

    # 2. Calculate n (count) for each leaf by summing the boolean masks.
    n = batch_masks.sum(dim=1, dtype=torch.float32)

    # 3. Use the masks to gather the relevant y values for sum and sum_sq.
    # We broadcast y_full and y_full.square() to the shape of the masks.
    # Where the mask is False, the value becomes 0 and does not contribute to the sum.
    y_broadcasted = y_full.expand_as(batch_masks)
    sum_y = torch.where(batch_masks, y_broadcasted, 0.0).sum(dim=1)

    y_sq_broadcasted = y_full.square().expand_as(batch_masks)
    sum_y_sq = torch.where(batch_masks, y_sq_broadcasted, 0.0).sum(dim=1)

    # Stack the results into the final [B, 3] tensor.
    return torch.stack([n, sum_y, sum_y_sq], dim=1)


def leaf_log_marginal(n: int, s1: float, s2: float, cfg: RewardConfig) -> float:
    """
    Calculates the log marginal likelihood of a leaf given its sufficient statistics.
    This is the core of the Bayesian reward function.
    """
    if n == 0:
        return 0.0

    # Posterior hyperparameters
    kappa_n = cfg.kappa0 + n
    nu_n = cfg.nu0 + n
    mu_n = (cfg.kappa0 * cfg.mu0 + s1) / kappa_n
    sigma_n_sq = (
        cfg.nu0 * cfg.sigma0_sq + s2 + cfg.kappa0 * cfg.mu0**2 - kappa_n * mu_n**2
    )

    # The marginal likelihood can be negative if the data is highly improbable
    # under the prior. We return -inf to signify an invalid state.
    if sigma_n_sq <= 1e-9: # Add a small epsilon for numerical stability
        return -float('inf')

    # Log of the marginal likelihood p(D_leaf | T)
    log_lik = (
        0.5 * torch.log(torch.tensor(cfg.kappa0 / kappa_n))
        + torch.lgamma(torch.tensor(nu_n / 2.0))
        - torch.lgamma(torch.tensor(cfg.nu0 / 2.0))
        + (cfg.nu0 / 2.0) * torch.log(torch.tensor(cfg.sigma0_sq))
        - (nu_n / 2.0) * torch.log(torch.tensor(sigma_n_sq))
    )
    return log_lik.item()


def partial_tree_log_reward(
    pt: PartialTree,
    X_bin_full: torch.Tensor,
    y_full: torch.Tensor,
    cfg: RewardConfig
) -> float:
    """
    Calculates the log-reward for a single partial tree (a GFN state).
    The reward is the marginal likelihood of the data in that leaf minus a
    complexity penalty based on the path length.
    """
    stats_tensor = batch_leaf_stats([pt], X_bin_full, y_full)[0]
    n, s1, s2 = stats_tensor[0].item(), stats_tensor[1].item(), stats_tensor[2].item()
    
    marginal_lik = leaf_log_marginal(int(n), s1, s2, cfg)
    complexity_penalty = cfg.beta * pt.n_nodes
    
    return marginal_lik - complexity_penalty


def tree_log_reward(
    tree: Tree,
    X_bin_full: torch.Tensor,
    y_full: torch.Tensor,
    cfg: RewardConfig
) -> float:
    """
    Calculates the total log-reward for a complete, multi-leaf tree.
    This is the sum of log marginal likelihoods over all leaves, minus a
    single complexity penalty for the entire tree structure.
    """
    # 1. Compute stats for all leaves in the tree in one batch
    stats = batch_leaf_stats(list(tree.leaves), X_bin_full, y_full)
    
    # 2. Calculate the log marginal likelihood for each leaf
    total_marginal_lik = 0.0
    for i in range(stats.size(0)):
        n, s1, s2 = stats[i, 0].item(), stats[i, 1].item(), stats[i, 2].item()
        total_marginal_lik += leaf_log_marginal(int(n), s1, s2, cfg)

    # 3. Calculate the complexity penalty based on the number of unique nodes
    unique_rules = set()
    for leaf in tree.leaves:
        unique_rules.update(leaf.rules)
    
    complexity_penalty = cfg.beta * len(unique_rules)
    
    return total_marginal_lik - complexity_penalty