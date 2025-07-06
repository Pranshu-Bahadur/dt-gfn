"""trainer.py – Core DT-GFN trainer with stabilized loss and per-epoch scoring.
==============================================================================
This module implements the main training and inference logic for DT-GFN.

Key Enhancements:
- Stabilized Trajectory Balance (TB) loss with a learnable log-partition
  function (log_z) and robust reward normalization.
- Live correlation scoring on the training set after each epoch.
- Correct reward integration using exact leaf statistics.
- Fully vectorized `_calculate_tb_loss` for high performance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from .tokenizer import Tokenizer
from .tree import DecisionRule, PartialTree, Tree
from .reward import RewardConfig, partial_tree_log_reward

autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainerConfig:
    """Configuration for the DT-GFN Trainer."""
    hidden_dim: int = 256
    depth_limit: int = 8
    epochs: int = 50
    batch_size: int = 1024
    replay_size: int = 16384
    lr: float = 3e-4
    epsilon: float = 0.1
    rollouts_per_epoch: int = 4096
    device: str = autocast_device

class PolicyNet(nn.Module):
    """Predicts next-action logits and a stop logit from a sequence of tokens."""
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.action_head = nn.Linear(hidden_dim, vocab_size)
        self.stop_head = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tokens)
        lstm_out, _ = self.lstm(embedded)
        last_hidden_state = lstm_out[:, -1, :]
        action_logits = self.action_head(last_hidden_state)
        stop_logit = self.stop_head(last_hidden_state).squeeze(-1)
        return action_logits, stop_logit

@dataclass(slots=True)
class ReplaySample:
    """A single state-action-reward transition stored in the replay buffer."""
    state_tokens: torch.Tensor
    action_token: int
    log_reward: float
    is_terminal: bool

class ReplayBuffer:
    """A simple circular buffer for storing off-policy experience."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[ReplaySample] = []
        self.position = 0

    def add(self, sample: ReplaySample):
        if len(self.memory) < self.capacity:
            self.memory.append(sample)
        else:
            self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[ReplaySample]:
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self) -> int:
        return len(self.memory)

class DTGFNTrainer:
    def __init__(
        self,
        X_bin: torch.Tensor,
        y: torch.Tensor,
        tokenizer: Tokenizer,
        reward_cfg: RewardConfig,
        t_cfg: TrainerConfig | None = None,
    ):
        self.cfg = t_cfg or TrainerConfig()
        self.device = torch.device(self.cfg.device)
        self.X_bin = X_bin.to(self.device)
        self.y = y.to(self.device)
        self.tok = tokenizer
        self.rcfg = reward_cfg
        
        self.policy = PolicyNet(len(self.tok), self.cfg.hidden_dim).to(self.device)
        # CRUCIAL FIX: Add a learnable log-partition function `log_z`
        self.log_z = nn.Parameter(torch.zeros(1, device=self.device))
        
        # Combine parameters for the optimizer
        params = list(self.policy.parameters()) + [self.log_z]
        self.optimizer = AdamW(params, lr=self.cfg.lr)
        
        self.buffer = ReplayBuffer(self.cfg.replay_size)
        self.trained_ensemble: List[Tree] | None = None

    def _sample_action(self, pt: PartialTree, is_feature_step: bool, use_epsilon: bool = True) -> int:
        """Samples the next action token using epsilon-greedy strategy."""
        if use_epsilon and (random.random() < self.cfg.epsilon or pt.is_terminal()):
            return self.tok.EOS

        with torch.no_grad():
            tokens = pt.to_tokens(self.tok, add_special=True).unsqueeze(0).to(self.device)
            action_logits, stop_logit = self.policy(tokens)
            
            valid_mask = torch.zeros_like(action_logits, dtype=torch.bool)
            if is_feature_step:
                valid_mask[:, self.tok.v.feature_start:self.tok.v.threshold_start] = True
            else:
                valid_mask[:, self.tok.v.threshold_start:self.tok.v.leaf_start] = True
            
            masked_logits = action_logits.masked_fill(~valid_mask, -float('inf'))
            all_logits = torch.cat([stop_logit.unsqueeze(-1), masked_logits], dim=-1)
            probs = torch.softmax(all_logits, dim=-1)
            
            action_idx = torch.multinomial(probs, 1).item()
            return self.tok.EOS if action_idx == 0 else (action_idx - 1)

    def _rollout(self):
        """Performs a single trajectory rollout to generate training samples."""
        pt = PartialTree(rules=(), depth_limit=self.cfg.depth_limit)
        
        while not pt.is_terminal():
            feature_token = self._sample_action(pt, is_feature_step=True)
            log_r = partial_tree_log_reward(pt, self.X_bin, self.y, self.rcfg)
            self.buffer.add(ReplaySample(pt.to_tokens(self.tok, add_special=True), feature_token, log_r, pt.is_terminal()))
            if feature_token == self.tok.EOS: break
            
            threshold_token = self._sample_action(pt, is_feature_step=False)
            self.buffer.add(ReplaySample(pt.to_tokens(self.tok, add_special=True), threshold_token, log_r, pt.is_terminal()))
            if threshold_token == self.tok.EOS: break
            
            feature_idx = self.tok.decode_one(feature_token)[1]
            threshold_idx = self.tok.decode_one(threshold_token)[1]
            rule = DecisionRule(feature=feature_idx, threshold_id=threshold_idx)
            pt = PartialTree(rules=pt.rules + (rule,), depth_limit=self.cfg.depth_limit)

    def train(self, num_ensemble_trees=200, leaves_per_tree=16):
        """Main training loop with per-epoch correlation scoring."""
        pbar = tqdm(range(self.cfg.epochs), desc="Epoch", leave=True)
        for epoch in pbar:
            self.policy.train()
            for _ in range(self.cfg.rollouts_per_epoch):
                self._rollout()
            
            if len(self.buffer) < self.cfg.batch_size:
                continue

            batch = self.buffer.sample(self.cfg.batch_size)
            loss = self._calculate_tb_loss(batch)
            
            # Stability check before backpropagation
            if torch.isfinite(loss):
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
            
            self.policy.eval()
            train_corr = self.score_correlation(num_probe_trees=10, leaves_per_tree=leaves_per_tree)
            pbar.set_postfix({"TB Loss": f"{loss.item():.4f}", "Train Corr": f"{train_corr:.4f}"})
        
        self.trained_ensemble = self.sample_ensemble(num_ensemble_trees, leaves_per_tree)

    def _calculate_tb_loss(self, batch: Sequence[ReplaySample]) -> torch.Tensor:
        """Calculates the stabilized Trajectory Balance loss."""
        
        max_len = max(s.state_tokens.size(0) for s in batch) if batch else 1
        states = torch.full((len(batch), max_len), self.tok.PAD, dtype=torch.long, device=self.device)
        for i, s in enumerate(batch):
            states[i, :s.state_tokens.size(0)] = s.state_tokens

        actions = torch.tensor([s.action_token for s in batch], dtype=torch.long, device=self.device)
        log_rewards = torch.tensor([s.log_reward for s in batch], dtype=torch.float32, device=self.device)

        # CRUCIAL FIX: Normalize log_rewards to be zero-mean, unit-variance.
        # This prevents the scale of rewards from dominating the loss.
        if len(log_rewards) > 1:
            log_rewards = (log_rewards - log_rewards.mean()) / (log_rewards.std() + 1e-8)

        action_logits, stop_logits = self.policy(states)
        
        all_logits = torch.cat([stop_logits.unsqueeze(-1), action_logits], dim=-1)
        log_probs = torch.log_softmax(all_logits, dim=-1)
        action_indices = torch.where(actions == self.tok.EOS, 0, actions + 1).unsqueeze(-1)
        log_pf = torch.gather(log_probs, 1, action_indices).squeeze(-1)
        
        # A safer log_pb calculation
        num_parents = (states != self.tok.PAD).sum(dim=1).float()
        log_pb = -torch.log(num_parents.clamp(min=1.0))
        
        # CRUCIAL FIX: Use the full Trajectory Balance objective with the learnable log_z
        squared_error = (self.log_z + log_pf - log_rewards - log_pb).pow(2)
        
        return squared_error.mean()

    def sample_ensemble(self, num_trees: int, leaves_per_tree: int) -> List[Tree]:
        """Samples an ensemble of full trees from the trained policy."""
        self.policy.eval()
        ensemble: List[Tree] = []
        pbar = tqdm(range(num_trees), desc="Sampling Final Ensemble", leave=False)
        for _ in pbar:
            leaves: List[PartialTree] = []
            for _ in range(leaves_per_tree):
                pt = PartialTree(rules=(), depth_limit=self.cfg.depth_limit)
                while not pt.is_terminal():
                    feature_token = self._sample_action(pt, is_feature_step=True, use_epsilon=False)
                    if feature_token == self.tok.EOS: break
                    threshold_token = self._sample_action(pt, is_feature_step=False, use_epsilon=False)
                    if threshold_token == self.tok.EOS: break

                    feature_idx = self.tok.decode_one(feature_token)[1]
                    threshold_idx = self.tok.decode_one(threshold_token)[1]
                    rule = DecisionRule(feature=feature_idx, threshold_id=threshold_idx)
                    pt = PartialTree(rules=pt.rules + (rule,), depth_limit=self.cfg.depth_limit)
                leaves.append(pt)
            
            if leaves:
                unique_leaves = tuple(sorted(list(set(leaves)), key=lambda p: str(p.rules)))
                ensemble.append(Tree(leaves=unique_leaves, weight=1.0))
        return ensemble

    def _predict_tree(self, tree: Tree, X_data: torch.Tensor) -> torch.Tensor:
        """Makes predictions for a single tree on a given dataset."""
        leaf_values = []
        for leaf in tree.leaves:
            train_mask = leaf.apply(self.X_bin)
            leaf_values.append(self.y[train_mask].mean() if train_mask.any() else self.y.mean())
        
        preds = torch.zeros(X_data.size(0), device=self.device)
        for mask, value in zip([leaf.apply(X_data) for leaf in tree.leaves], leaf_values):
            preds[mask] = value
        return preds * tree.weight

    def score_correlation(self, num_probe_trees: int, leaves_per_tree: int) -> float:
        """Calculates Pearson correlation of a small probe ensemble on the training data."""
        probe_ensemble = self.sample_ensemble(num_probe_trees, leaves_per_tree)
        if not probe_ensemble: return 0.0

        agg_preds = torch.zeros_like(self.y, dtype=torch.float32)
        for tree in probe_ensemble:
            agg_preds += self._predict_tree(tree, self.X_bin)

        preds_mean, preds_std = agg_preds.mean(), agg_preds.std()
        target_mean, target_std = self.y.mean(), self.y.std()

        if preds_std < 1e-8 or target_std < 1e-8: return 0.0

        preds_z = (agg_preds - preds_mean) / preds_std
        target_z = (self.y - target_mean) / target_std
        
        return torch.dot(preds_z, target_z).item() / len(self.y)