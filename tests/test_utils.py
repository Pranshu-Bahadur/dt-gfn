import random
import pytest
import torch

from src.utils import (
    sequence_to_predictor,
    ReplayBuffer,
    tb_loss,
    fl_loss,
    _safe_sample,
)
from src.tokenizer import Tokenizer


def test_sequence_to_predictor_empty_and_leaf():
    # Setup tokenizer for 1 feature, 2 bins
    tok = Tokenizer(num_features=1, num_bins=2)
    # Trajectory with no actions: only BOS and EOS
    traj_empty = [tok.BOS, tok.EOS]
    X = torch.tensor([[0]], dtype=torch.int64)
    y = torch.tensor([5.0])
    pred_fn_empty = sequence_to_predictor(traj_empty, X, y, tok)
    # Expect zeros because no structure
    out_empty = pred_fn_empty(X)
    assert torch.allclose(out_empty, torch.zeros(1))

    # Trajectory: BOS, F0, TH0, LEAF, LEAF, EOS
    traj = [
        tok.BOS,
        tok.encode_feature(0),
        tok.encode_threshold(0),
        tok.encode_leaf(0),
        tok.encode_leaf(0),
        tok.EOS,
    ]
    X_single = torch.tensor([[0]], dtype=torch.int64)
    y_single = torch.tensor([10.0])
    pred_fn = sequence_to_predictor(traj, X_single, y_single, tok)
    out = pred_fn(X_single)
    # The only leaf receives the y_mean = 10
    assert torch.allclose(out, torch.tensor([10.0]))


def test_sequence_to_predictor_inference_defaults_to_zero():
    tok = Tokenizer(num_features=1, num_bins=2)
    traj = [
        tok.BOS,
        tok.encode_feature(0),
        tok.encode_threshold(1),
        tok.encode_leaf(0),
        tok.encode_leaf(0),
        tok.EOS,
    ]
    X = torch.tensor([[1]], dtype=torch.int64)
    pred_fn = sequence_to_predictor(traj, X, None, tok)
    out = pred_fn(X)
    # No y_target => default leaf value = 0
    assert torch.allclose(out, torch.zeros_like(out))


def test_replay_buffer_capacity_and_sampling():
    buf = ReplayBuffer(capacity=3)
    # Add 5 dummy items with ascending reward
    for i in range(5):
        buf.add(reward=float(i), traj=[i], prior=0.0, idxs=torch.tensor([i]))
    # Only top 3 by reward (4,3,2) should remain
    rewards = [item[0] for item in buf.data]
    assert rewards == [4.0, 3.0, 2.0]
    # Sampling more than length returns full buffer
    assert len(buf.sample(10)) == 3
    # Sampling fewer returns that many
    assert len(buf.sample(2)) == 2


def test_tb_loss_zero_when_balanced():
    log_pf = torch.zeros((2, 3))
    log_pb = torch.zeros((2, 3))
    log_z = torch.zeros((), requires_grad=True)
    R = torch.ones((2,))
    prior = torch.zeros((2,))
    loss = tb_loss(log_pf, log_pb, log_z, R, prior)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_fl_loss_zero_when_matching():
    logF = torch.zeros((2, 2))
    log_pf = torch.zeros((2, 1))
    log_pb = torch.zeros((2, 1))
    dE = torch.zeros((2, 1))
    loss = fl_loss(logF, log_pf, log_pb, dE)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_safe_sample_respects_mask_and_temperature():
    torch.manual_seed(0)
    logits = torch.tensor([0.1, 2.0, 0.5])
    mask = torch.tensor([False, True, False])
    # Only index 1 is allowed
    idx = _safe_sample(logits, mask, temperature=1.0)
    assert idx == 1

    # If multiple allowed, distribution is non-trivial
    mask2 = torch.tensor([True, True, False])
    counts = {0: 0, 1: 0}
    for _ in range(1000):
        k = _safe_sample(logits, mask2, temperature=0.5)
        assert k in (0, 1)
        counts[k] += 1
    assert counts[0] > 0 and counts[1] > 0
