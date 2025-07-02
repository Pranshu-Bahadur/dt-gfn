import pytest

from src.tokenizer import Vocab, Tokenizer, TokenType


test_vocab_params = {
    "num_features": 4,
    "num_thresholds": 6,
    "num_leaves": 3,
}

@ pytest.fixture
def vocab_and_tok():
    v = Vocab(**test_vocab_params)
    tok = Tokenizer(vocab=v)
    return v, tok


def test_vocab_size_and_len(vocab_and_tok):
    v, _ = vocab_and_tok
    expected = (v.EOS + 1) + v.num_features + v.num_thresholds + v.num_leaves
    assert v.size() == expected
    assert len(v) == expected


def test_encode_ids(vocab_and_tok):
    v, tok = vocab_and_tok
    # Feature tokens
    for i in range(v.num_features):
        tid = tok.encode_feature(i)
        assert tid == v.split_start + i
    # Threshold tokens
    for i in range(v.num_thresholds):
        tid = tok.encode_threshold(i)
        assert tid == v.split_start + v.num_features + i
    # Leaf tokens
    for i in range(v.num_leaves):
        tid = tok.encode_leaf(i)
        assert tid == v.split_start + v.num_features + v.num_thresholds + i


def test_decode_one_and_decode(vocab_and_tok):
    v, tok = vocab_and_tok
    # Build a mixed list: [PAD, BOS, EOS, some tokens]
    tokens = [v.PAD, v.BOS, v.EOS]
    # Pick one of each category
    f_idx = 1
    t_idx = 2
    l_idx = 0
    tokens += [
        tok.encode_feature(f_idx),
        tok.encode_threshold(t_idx),
        tok.encode_leaf(l_idx)
    ]
    # Test decode_one on non-special IDs
    t1 = tok.decode_one(tok.encode_feature(f_idx))
    assert t1 == (TokenType.FEATURE, f_idx)
    t2 = tok.decode_one(tok.encode_threshold(t_idx))
    assert t2 == (TokenType.THRESHOLD, t_idx)
    t3 = tok.decode_one(tok.encode_leaf(l_idx))
    assert t3 == (TokenType.LEAF, l_idx)
    # Test decode skips special tokens and yields correct sequence
    decoded = tok.decode(tokens)
    assert decoded == [
        (TokenType.FEATURE, f_idx),
        (TokenType.THRESHOLD, t_idx),
        (TokenType.LEAF, l_idx)
    ]


def test_invalid_decode_one_raises(vocab_and_tok):
    v, tok = vocab_and_tok
    # Below split_start
    with pytest.raises(ValueError):
        tok.decode_one(v.PAD)
    # Above valid range
    with pytest.raises(ValueError):
        tok.decode_one(v.size())


def test_invalid_encode_range_raises(vocab_and_tok):
    v, tok = vocab_and_tok
    # Feature index out of range
    with pytest.raises(IndexError):
        tok.encode_feature(-1)
    with pytest.raises(IndexError):
        tok.encode_feature(v.num_features)
    # Threshold index out of range
    with pytest.raises(IndexError):
        tok.encode_threshold(-1)
    with pytest.raises(IndexError):
        tok.encode_threshold(v.num_thresholds)
    # Leaf index out of range
    with pytest.raises(IndexError):
        tok.encode_leaf(-1)
    with pytest.raises(IndexError):
        tok.encode_leaf(v.num_leaves)

