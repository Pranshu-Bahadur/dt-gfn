# src/trainer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import random
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from src.binning import BinConfig, Binner
from src.env import TabularEnv
from src.tokenizer import Tokenizer
from src.policy import PolicyPaperMLP, PolicyTransformer
from src.utils import (
    PrioritizedReplayBuffer,  # <-- Use PrioritizedReplayBuffer
    ExplorationScheduler,   # <-- Import ExplorationScheduler
    tb_loss,
    fl_loss,
    _safe_sample,
    sequence_to_predictor,
    dE_split_gain
)


# --------------------------------------------------------------------------- #
# 1.  Hyper-parameters
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    feature_cols: List[str]
    target_col: str = "label"
    n_bins: int = 256
    bin_strategy: str = "feature_quantile"
    device: str = "cuda"

    # Boost-GFN
    updates: int = 50
    rollouts: int = 60
    batch_size: int = 8_192
    max_depth: int = 4
    top_k_trees: int = 10
    boosting_lr: float = 0.1

    # Policy net
    lstm_hidden: int = 512
    mlp_layers: int = 8
    mlp_width: int = 256
    lr: float = 5e-5

    # Exploration Strategy
    initial_epsilon: float = 1.0
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.99


# --------------------------------------------------------------------------- #
# 2.  Trainer (fit + predict)
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer: Optional[Tokenizer] = None
        self.binner: Optional[Binner] = None
        self.ensemble: list[list[list[int]]] = []
        self.y_mean: float = 0.0

    # --------------------------------------------------------------------- #
    # fit
    # --------------------------------------------------------------------- #
    def fit(self, df_train: pd.DataFrame) -> "Trainer":
        c = self.cfg
        device = c.device

        # ---- Tokenizer & Binner ------------------------------------------
        self.tokenizer = Tokenizer(
            num_features=len(c.feature_cols),
            num_bins=c.n_bins,
            num_leaves=2**c.max_depth - 1,
        )
        bin_cfg = BinConfig(c.n_bins, c.bin_strategy)
        self.binner = Binner(bin_cfg)

        # ---- Environment --------------------------------------------------
        env = TabularEnv(
            df_train,
            feature_cols=c.feature_cols,
            target_col=c.target_col,
            bin_config=bin_cfg,
            tokenizer=self.tokenizer,
            device=device,
        )

        self.binner = env.binner

        # ---- Policy Nets --------------------------------------------------
        vocab_size = len(self.tokenizer)
        #pf = PolicyTransformer(vocab_size=len(self.tokenizer), d_model=256,
        #               n_layers=12, n_heads=4, d_ff=512).to(device)
        #pb = PolicyTransformer(vocab_size=len(self.tokenizer), d_model=256,
        #               n_layers=12, n_heads=4, d_ff=512).to(device)
        pf = PolicyPaperMLP(vocab_size, c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        pb = PolicyPaperMLP(vocab_size, c.lstm_hidden, c.mlp_layers, c.mlp_width).to(device)
        pf, pb = torch.jit.script(pf), torch.jit.script(pb)

        log_z = torch.zeros((), device=device, requires_grad=True)
        opt_pf = torch.optim.AdamW(pf.parameters(), lr=c.lr)
        opt_pb = torch.optim.AdamW(pb.parameters(), lr=c.lr)
        opt_z = torch.optim.Adam([log_z], lr=c.lr / 10)
        optimizers = [opt_pf, opt_pb, opt_z]

        # ---- LR Schedulers ------------------------------------------------
        warm, total = 10, c.updates
        scheds = []
        for opt in optimizers:
            scheds.append(
                SequentialLR(
                    opt,
                    [
                        LambdaLR(opt, lambda it: min(1.0, it / warm)),
                        CosineAnnealingLR(opt, T_max=max(1, total - warm)),
                    ],
                    milestones=[warm],
                )
            )

        # ---- Replay Buffer, Scheduler, & Baseline -------------------------
        buf = PrioritizedReplayBuffer(capacity=10_000)
        scheduler = ExplorationScheduler(
            c.initial_epsilon, c.min_epsilon, c.epsilon_decay
        )
        base_pred = torch.full_like(env.y_full, env.y_full.mean())
        self.y_mean = env.y_full.mean().item()

        # ---- Pre-compute ID ranges for masks -----------------------------
        FEAT_START = 3
        TH_START   = FEAT_START + self.tokenizer.num_features
        LEAF_START = TH_START + self.tokenizer.num_bins

        # ------------------------------------------------------------------ #
        # Boost loop
        # ------------------------------------------------------------------ #
        for upd in range(1, c.updates + 1):
            residuals = env.y_full - base_pred
            env.y = residuals.clone()

            tb_acc, fl_acc, completed = 0.0, 0.0, 0
            for opt in optimizers:
                opt.zero_grad()

            pbar = tqdm(range(c.rollouts), desc=f"Upd {upd:02d}", leave=False)
            epsilon = scheduler.get_epsilon(upd)
            pbar.set_postfix({"eps": f"{epsilon:.3f}"})

            for _ in pbar:
                env.reset(c.batch_size)
                seq_ids = [self.tokenizer.BOS]
                open_depths = [0]

                # -------- Rollout Sampling with Dynamic Exploration --------
                with torch.no_grad():
                    while not env.done and open_depths:
                        x = torch.tensor([seq_ids], device=device)
                        logits, _ = pf(x)
                        logits = logits[0, -1]

                        mask = torch.zeros_like(logits, dtype=torch.bool)
                        cur_depth = open_depths[-1]

                        if env.open_leaves > 0 and cur_depth < c.max_depth:
                            mask[FEAT_START:TH_START] = True
                        if env.open_leaves > 0:
                            mask[LEAF_START:] = True

                        if not mask.any():
                            break

                        # Use random actions for exploration
                        if random.random() < epsilon:
                            # Create a tensor of possible actions from the mask
                            possible_actions = torch.where(mask)[0]
                            if len(possible_actions) > 0:
                                tok_id = random.choice(possible_actions).item()
                            else:
                                break # No valid actions
                        else:
                            tok_id = _safe_sample(logits, mask, temperature=1.0)

                        kind, idx = self.tokenizer.decode_one(tok_id)
                        env.step((kind, idx))
                        seq_ids.append(tok_id)

                        if kind == "feature":
                            logits2, _ = pf(torch.tensor([seq_ids], device=device))
                            logits_th = logits2[0, -1]
                            th_mask = torch.zeros_like(logits_th, dtype=torch.bool)
                            th_mask[TH_START:LEAF_START] = True
                            th_id = _safe_sample(logits_th, th_mask, temperature=1.0)
                            env.step(self.tokenizer.decode_one(th_id))
                            seq_ids.append(th_id)

                            d = open_depths.pop()
                            open_depths.extend([d + 1, d + 1])
                        else:
                            open_depths.pop()

                seq_ids.append(self.tokenizer.EOS)
                if env.open_leaves != 0:
                    continue

                # Reward & prior
                R, prior, _, _ = env.evaluate(current_beta=0.0)
                R = torch.tensor([R], device=device)
                completed += 1

                # --- Losses ---
                fwd = torch.tensor([seq_ids], device=device)
                log_pf = pf.log_prob(fwd)
                logF = pf.log_F(fwd)

                bwd = torch.flip(fwd, dims=[1])
                log_pb = pb.log_prob(bwd)

                dE = dE_split_gain(fwd, self.tokenizer, env)

                l_tb = tb_loss(log_pf, log_pb, log_z, R.unsqueeze(0), prior)
                l_fl = fl_loss(logF, log_pf, log_pb, dE)
                (l_tb + 0.1 * l_fl).backward()
                tb_acc += l_tb.item()
                fl_acc += l_fl.item()

                buf.add(R.item(), seq_ids, prior.item(), env.idxs.clone())

            # Optimise
            if completed:
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]["params"], 1.0)
                    opt.step()
                for sch in scheds:
                    sch.step()

            # ---- Boosting Update ----
            if len(buf) > 0:
                best = buf.sample(c.top_k_trees) # Sample best trees from prioritized buffer
                top_seqs = [t[1] for t in best]
                predictors = [
                    sequence_to_predictor(s, env.X_full, residuals, self.tokenizer)
                    for s in top_seqs
                ]
                self.ensemble.append(predictors)

                step_pred = torch.zeros_like(base_pred)
                for s in top_seqs:
                    f = sequence_to_predictor(s, env.X_full, residuals, self.tokenizer)
                    step_pred += f(env.X_full)
                step_pred /= len(top_seqs)
                base_pred += c.boosting_lr * step_pred

                rho = torch.corrcoef(torch.stack([base_pred, env.y_full]))[0, 1].item()
                print(
                    f"Update {upd:02d} | comps {completed}/{c.rollouts} "
                    f"| TB {tb_acc / max(1, completed):.4f} "
                    f"| FL {fl_acc / max(1, completed):.4f} "
                    f"| ρ_train {rho:+.3f}"
                )
                buf.buffer.clear() # Clear the buffer for the next boosting iteration

        return self

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("Call fit() first.")

        device = self.cfg.device

        if self.binner is None:
            X_arr = df[self.cfg.feature_cols].values.astype("int8")
        else:
            X_arr = self.binner.transform(df[self.cfg.feature_cols]).values.astype("int8")
        X = torch.tensor(X_arr, dtype=torch.int8, device=device)

        preds = torch.full((len(df),), self.y_mean, dtype=torch.float32, device=device)
        for pred_fns in self.ensemble:
            step = torch.zeros_like(preds)
            for fn in pred_fns:
                step += fn(X)
            preds += self.cfg.boosting_lr * step / max(1, len(pred_fns))

        return preds.cpu().numpy()

    # ------------------------------------------------------------------ #
    # sklearn helpers
    # ------------------------------------------------------------------ #
    def get_params(self, deep=True):
        return asdict(self.cfg)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.cfg, k, v)
        return self