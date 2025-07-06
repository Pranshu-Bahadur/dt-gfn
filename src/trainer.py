"""trainer.py – Core DT‑GFN trainer (library mode, multi‑tree ensemble)
====================================================================
A **library‑style** DT‑GFN implementation that learns a GFlowNet over
partial tree paths and can *sample* full decision trees for inference on
Numerai regression targets.

Key points
----------
* Gaussian‑NIG closed‑form reward (see `reward.py`).
* Off‑policy Trajectory Balance with replay buffer.
* **True tree inference** – each tree routes every row to exactly one
  leaf and uses the leaf’s posterior mean; an ensemble averages across
  trees.

This file intentionally omits a CLI; import `DTGFNTrainer` in your own
wrapper (e.g. a scikit‑learn `Estimator`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .binner import Binner
from .tokenizer import Tokenizer
from .tree import DecisionRule, PartialTree, Tree
from .reward import (
    RewardConfig,
    precompute_stats,
    partial_tree_log_reward,
)

# ---------------------------------------------------------------------------
# 0.  Config dataclasses
# ---------------------------------------------------------------------------

autocast_device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class TrainerConfig:
    hidden_dim: int = 128
    depth_limit: int = 6
    epochs: int = 3
    batch_size: int = 1024  # gradient steps per epoch = replay_size / batch_size
    replay_size: int = 4096
    lr: float = 1e-3
    epsilon: float = 0.2  # exploration prob for ε‑greedy
    leaves_per_tree: int = 8
    device: str = autocast_device


# ---------------------------------------------------------------------------
# 1.  Policy network
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """Tiny 2‑head MLP that predicts next‑action logits + stop logit."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.rule_head = nn.Linear(hidden_dim, vocab_size)  # for choosing next token
        self.stop_head = nn.Linear(hidden_dim, 1)  # for deciding to terminate

    def forward(self, tokens: torch.LongTensor):
        # tokens: [B, L]
        x = self.embed(tokens).mean(dim=1)  # simple bag‑of‑tokens pooling
        h = self.body(x)
        return self.rule_head(h), self.stop_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 2.  Replay buffer util
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ReplaySample:
    state_tokens: torch.LongTensor  # [L]
    action_token: int  # the token chosen at this state
    log_reward: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.mem: List[ReplaySample] = []

    def add(self, sample: ReplaySample):
        if len(self.mem) >= self.capacity:
            self.mem.pop(0)
        self.mem.append(sample)

    def sample_batch(self, batch_size: int) -> List[ReplaySample]:
        return random.sample(self.mem, min(batch_size, len(self.mem)))


# ---------------------------------------------------------------------------
# 3.  Main trainer class
# ---------------------------------------------------------------------------

class DTGFNTrainer:
    def __init__(
        self,
        X_bin: torch.LongTensor,
        y: torch.Tensor,
        tokenizer: Tokenizer,
        reward_cfg: RewardConfig,
        t_cfg: TrainerConfig | None = None,
    ) -> None:
        self.X_bin = X_bin.to(reward_cfg.device)
        self.y = y.to(reward_cfg.device)
        self.tok = tokenizer
        self.rcfg = reward_cfg
        self.cfg = t_cfg or TrainerConfig()
        self.device = torch.device(self.cfg.device)

        self.stats = precompute_stats(self.X_bin, self.y, self.tok.n_bins())
        self.policy = PolicyNet(self.tok.vocab_size, self.cfg.hidden_dim).to(self.device)
        self.opt = AdamW(self.policy.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.replay_size)

    # --------------------------- rollouts --------------------------- #

    def _sample_action(self, pt: PartialTree) -> int:
        """ε‑greedy pick of next token (or EOS)."""
        if random.random() < self.cfg.epsilon or pt.is_terminal(self.cfg.depth_limit):
            return self.tok.eos_id
        # model‑based
        tokens = pt.to_tokens(self.tok).unsqueeze(0).to(self.device)
        rule_logits, stop_logit = self.policy(tokens)
        # concat rule+stop into one categorical distribution
        logits = torch.cat([stop_logit, rule_logits.squeeze(0)], dim=0)  # [1+V]
        probs = torch.softmax(logits, dim=0)
        action_idx = torch.multinomial(probs, 1).item()
        if action_idx == 0:
            return self.tok.eos_id
        return action_idx - 1  # shift because of stop logit at 0

    def rollout(self):
        pt = PartialTree(())
        while True:
            a_token = self._sample_action(pt)
            log_r = partial_tree_log_reward(pt, self.stats, self.rcfg)
            self.buffer.add(
                ReplaySample(state_tokens=pt.to_tokens(self.tok), action_token=a_token, log_reward=log_r)
            )
            if a_token == self.tok.eos_id or pt.is_terminal(self.cfg.depth_limit):
                break
            # decode action token to feature,threshold tuple
            f_id, thr_id = self.tok.decode_rule(a_token)
            pt = PartialTree(pt.rules + (DecisionRule(f_id, thr_id),))

    # --------------------------- training --------------------------- #

    def train(self):
        pbar = tqdm(range(self.cfg.epochs), desc="epoch", position=0)
        for _ in pbar:
            # generate new rollouts
            for _ in range(self.cfg.batch_size):
                self.rollout()
            # gradient steps
            batch = self.buffer.sample_batch(self.cfg.batch_size)
            if not batch:
                continue
            loss = self._tb_loss(batch)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            pbar.set_postfix({"TB_loss": loss.item(), "corr": f"{self.score_corr():.4f}"})

    def _tb_loss(self, batch: Sequence[ReplaySample]):
        # simple TB: log P_F – log P_B – log R  => minimise squared error
        losses = []
        for sample in batch:
            tokens = sample.state_tokens.unsqueeze(0).to(self.device)
            rule_logits, stop_logit = self.policy(tokens)
            if sample.action_token == self.tok.eos_id:
                log_pf = stop_logit.squeeze(0)
            else:
                log_pf = rule_logits.squeeze(0)[sample.action_token]
            log_pb = -np.log(len(sample.state_tokens))  # uniform backward
            losses.append((log_pf + log_pb - sample.log_reward) ** 2)
        return torch.stack(losses).mean()

    # --------------------------- inference helpers --------------------------- #

    def _predict_tree(self, tree: Tree) -> torch.Tensor:
        """Vectorised prediction for *one* Tree on stored X_bin."""
        preds = torch.zeros_like(self.y)
        assigned = torch.zeros_like(self.y, dtype=torch.bool)
        for leaf in tree.leaves:
            mask = leaf.apply(self.X_bin)
            sum_y = self.y[mask].sum()
            n = mask.sum()
            mu = (sum_y / max(n, 1)).item() if n > 0 else 0.0
            preds[mask] = mu * tree.weight
            assigned |= mask
        # optional: if any rows not assigned (shouldn't happen), keep 0.0
        return preds

    def sample_ensemble(self, num_trees: int, leaves_per_tree: int | None = None) -> List[Tree]:
        leaves_per_tree = leaves_per_tree or self.cfg.leaves_per_tree
        ensemble: List[Tree] = []
        for _ in range(num_trees):
            leaves = []
            pt = PartialTree(())
            for _ in range(leaves_per_tree):
                # simple random rollout to get a leaf
                while not pt.is_terminal(self.cfg.depth_limit):
                    a_tok = self._sample_action(pt)
                    if a_tok == self.tok.eos_id:
                        break
                    f, thr = self.tok.decode_rule(a_tok)
                    pt = PartialTree(pt.rules + (DecisionRule(f, thr),))
                leaves.append(pt)
            ensemble.append(Tree(tuple(leaves), weight=1.0 / num_trees))
        return ensemble

    def score_corr(self, probe_trees: int = 5):
        """Returns Pearson correlation on the stored training set."""
        ensemble = self.sample_ensemble(num_trees=probe_trees)
        agg_preds = torch.zeros_like(self.y, dtype=torch.float32)
        for t in ensemble:
            agg_preds += self._predict_tree(t)
        # normalise: mean 0, std 1 to mimic Numerai rank corr≈
        preds = (agg_preds - agg_preds.mean()) / (agg_preds.std() + 1e-8)
        target = (self.y - self.y.mean()) / (self.y.std() + 1e-8)
        corr = torch.matmul(preds, target) / len(target)
        return corr.item()

