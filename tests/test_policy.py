import torch
import pytest
from src.policy import PolicyPaperMLP

torch.manual_seed(42)


def test_forward_shape_and_types():
    vocab_size = 10
    lstm_hidden = 8
    mlp_layers = 2
    mlp_width = 16
    policy = PolicyPaperMLP(vocab_size, lstm_hidden, mlp_layers, mlp_width)
    
    batch_size, seq_length = 3, 5
    seqs = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    logits, flow = policy.forward(seqs)
    
    # shapes
    assert logits.shape == (batch_size, seq_length, vocab_size)
    assert flow.shape == (batch_size, seq_length)
    # types
    assert logits.dtype == torch.float32
    assert flow.dtype == torch.float32


def test_log_prob_matches_manual_computation():
    vocab_size = 7
    policy = PolicyPaperMLP(vocab_size, lstm_hidden=4, mlp_layers=1, mlp_width=8)
    seqs = torch.randint(0, vocab_size, (2, 6))
    
    # manual log-prob calculation
    logits, _ = policy.forward(seqs[:, :-1])
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = torch.gather(log_probs, -1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # model’s log_prob
    actual = policy.log_prob(seqs)
    
    assert torch.allclose(actual, expected)


def test_log_prob_empty_for_short_sequences():
    policy = PolicyPaperMLP(vocab_size=5, lstm_hidden=4, mlp_layers=1, mlp_width=8)
    seqs = torch.randint(0, 5, (4, 1))
    lp = policy.log_prob(seqs)
    assert lp.shape == (4, 0)


def test_log_F_matches_forward():
    vocab_size = 6
    policy = PolicyPaperMLP(vocab_size, lstm_hidden=5, mlp_layers=2, mlp_width=8)
    seqs = torch.randint(0, vocab_size, (3, 7))
    
    _, flow_forward = policy.forward(seqs)
    flow_direct = policy.log_F(seqs)
    
    assert torch.allclose(flow_forward, flow_direct)
