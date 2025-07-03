# src/trainer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from src.binning import BinConfig, Binner
from src.env import TabularEnv
from src.tokenizer import Tokenizer
from src.trees.policy import PolicyPaperMLP
from src.utils import (
    ReplayBuffer,
    tb_loss,
    fl_loss,
    _safe_sample,
    sequence_to_predictor,
)


# -----------------------------------------------------------------------------#
# 1.  Hyper-parameter bundle
# -----------------------------------------------------------------------------#
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

    # Policy network
    lstm_hidden: int = 512
    mlp_layers: int = 8
    mlp_width: int = 256
    lr: float = 5e-5


# -----------------------------------------------------------------------------#
# 2.  Trainer  (scikit-learn style: fit + predict)
# -----------------------------------------------------------------------------#
class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Populated in fit()
        self.tokenizer: Optional[Tokenizer] = None
        self.binner: Optional[Binner] = None
        self.ensemble: list[list[list[int]]] = []
        self.y_mean: float = 0.0

    # ---------------------------------------------------------------------#
    # fit
    # ---------------------------------------------------------------------#
    def fit(self, df_train: pd.DataFrame) -> "Trainer":
        cfg = self.cfg
        device = cfg.device

        # --- Tokenizer & Binner ------------------------------------------------
        self.tokenizer = Tokenizer(
            num_features=len(cfg.feature_cols), num_bins=cfg.n_bins
        )
        bin_cfg = BinConfig(cfg.n_bins, cfg.bin_strategy)
        self.binner = Binner(bin_cfg)

        # --- Environment ------------------------------------------------------
        env = TabularEnv(
            df_train,
            feature_cols=cfg.feature_cols,
            target_col=cfg.target_col,
            bin_config=bin_cfg,
            tokenizer=self.tokenizer,
            device=device,
        )

        # --- Policy networks --------------------------------------------------
        vocab_size = self.tokenizer.vocab_size
        pf = PolicyPaperMLP(vocab_size, cfg.lstm_hidden, cfg.mlp_layers, cfg.mlp_width).to(device)
        pb = PolicyPaperMLP(vocab_size, cfg.lstm_hidden, cfg.mlp_layers, cfg.mlp_width).to(device)
        pf = torch.jit.script(pf)
        pb = torch.jit.script(pb)

        log_z = torch.zeros((), device=device, requires_grad=True)

        opt_pf = torch.optim.AdamW(pf.parameters(), lr=cfg.lr)
        opt_pb = torch.optim.AdamW(pb.parameters(), lr=cfg.lr)
        opt_z  = torch.optim.Adam([log_z], lr=cfg.lr / 10)
        optimizers = [opt_pf, opt_pb, opt_z]

        # Cosine schedule with warm-up
        warm, total = 10, cfg.updates
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

        # --- Replay buffer & bookkeeping --------------------------------------
        buf = ReplayBuffer(capacity=10_000)
        base_pred = torch.full_like(env.y_full, env.y_full.mean())
        self.y_mean = env.y_full.mean().item()

        # ------------------------------------------------------------------#
        # Main Boost Loop
        # ------------------------------------------------------------------#
        for upd in range(1, cfg.updates + 1):
            residuals = env.y_full - base_pred
            env.y = residuals.clone()

            tb_acc, fl_acc, completed = 0.0, 0.0, 0
            for opt in optimizers:
                opt.zero_grad()

            pbar = tqdm(range(cfg.rollouts), desc=f"Upd {upd:02d}", leave=False)
            for _ in pbar:
                env.reset(cfg.batch_size)
                seq_ids = [self.tokenizer.BOS]
                open_depths = [0]

                # ------------ rollout ------------
                with torch.no_grad():
                    while not env.done and open_depths:
                        x = torch.tensor([seq_ids], device=device)
                        logits, _ = pf(x)
                        logits = logits[0, -1]

                        mask = torch.zeros_like(logits, dtype=torch.bool)
                        cur_depth = open_depths[-1]
                        if env.open_leaves > 0 and cur_depth < cfg.max_depth:
                            # allow feature splits
                            mask[self.tokenizer.split_start:
                                 self.tokenizer.split_start + self.tokenizer.num_features] = True
                        if env.open_leaves > 0:
                            # allow leaf tokens
                            mask[self.tokenizer.split_start + self.tokenizer.num_features +
                                 self.tokenizer.num_bins:] = True

                        if not mask.any():
                            break

                        tok_id = _safe_sample(logits, mask, temperature=1.0)
                        kind, idx = self.tokenizer.decode_one(tok_id)
                        env.step((kind, idx))
                        seq_ids.append(tok_id)

                        # If feature, immediately sample threshold
                        if kind == "feature":
                            logits2, _ = pf(torch.tensor([seq_ids], device=device))
                            logits_th = logits2[0, -1]
                            th_mask = torch.zeros_like(logits_th, dtype=torch.bool)
                            th_mask[self.tokenizer.split_start + self.tokenizer.num_features:
                                    self.tokenizer.split_start + self.tokenizer.num_features + self.tokenizer.num_bins] = True
                            th_id = _safe_sample(logits_th, th_mask, temperature=1.0)
                            env.step(self.tokenizer.decode_one(th_id))
                            seq_ids.append(th_id)

                            d = open_depths.pop()
                            open_depths.extend([d + 1, d + 1])
                        else:
                            open_depths.pop()

                seq_ids.append(self.tokenizer.EOS)

                if env.open_leaves != 0:
                    continue  # incomplete tree

                # Evaluate reward & prior
                R, prior, _, _ = env.evaluate(current_beta=0.1)
                completed += 1

                # ----------------- losses -----------------
                fwd_tokens = torch.tensor([seq_ids], device=device)
                log_pf = pf.log_prob(fwd_tokens)
                logF = pf.log_F(fwd_tokens)

                bwd_tokens = torch.flip(fwd_tokens, dims=[1])
                log_pb = pb.log_prob(bwd_tokens)

                dE = torch.zeros_like(log_pf)  # placeholder (gain term optional)

                l_tb = tb_loss(log_pf, log_pb, log_z, R.unsqueeze(0), prior)
                l_fl = fl_loss(logF, log_pf, log_pb, dE)
                loss = l_tb + 0.1 * l_fl
                loss.backward()

                tb_acc += l_tb.item()
                fl_acc += l_fl.item()

                buf.add(R.item(), seq_ids, prior.item(), env.idxs.clone())

            # Gradient update
            if completed:
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
                    opt.step()
                for sch in scheds:
                    sch.step()

            # ---------------- Boosting update ----------------
            if buf.data:
                top_k = buf.data[: cfg.top_k_trees]
                top_sequences = [t[1] for t in top_k]
                self.ensemble.append(top_sequences)

                step_pred = torch.zeros_like(base_pred)
                for s in top_sequences:
                    pred_fn = sequence_to_predictor(
                        s,
                        env.X_full,
                        residuals,
                        self.tokenizer,
                    )
                    step_pred += pred_fn(env.X_full)

                step_pred /= len(top_sequences)
                base_pred += cfg.boosting_lr * step_pred

                corr = torch.corrcoef(torch.stack([base_pred, env.y_full]))[0, 1].item()
                print(
                    f"Update {upd:02d} | comps {completed}/{cfg.rollouts} "
                    f"| TB {tb_acc / max(1, completed):.4f} "
                    f"| FL {fl_acc / max(1, completed):.4f} "
                    f"| ρ_train {corr:+.3f}"
                )

                buf.data.clear()

        return self  # sklearn expects .fit to return self

    # ---------------------------------------------------------------------#
    # predict
    # ---------------------------------------------------------------------#
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("Call fit() first.")

        # Bin or cast features
        if self.binner is None:
            X_arr = df[self.cfg.feature_cols].values.astype("int8")
        else:
            X_df = self.binner.transform(df[self.cfg.feature_cols])
            X_arr = X_df.values.astype("int8")
        X = torch.tensor(X_arr, dtype=torch.int8)

        preds = torch.full((len(df),), self.y_mean, dtype=torch.float32)
        for seq_list in self.ensemble:
            step = torch.zeros_like(preds)
            for s in seq_list:
                f = sequence_to_predictor(s, X, None, self.tokenizer)
                step += f(X)
            preds += self.cfg.boosting_lr * step / max(1, len(seq_list))

        return preds.numpy()

    # ---------------------------------------------------------------------#
    # sklearn helpers
    # ---------------------------------------------------------------------#
    def get_params(self, deep: bool = True):
        return asdict(self.cfg)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self.cfg, k, v)
        return self

