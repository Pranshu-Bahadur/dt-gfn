import pytest

from src.tokenizer import Tokenizer

test_params = {
    "num_features": 5,
    "num_bins": 7,
    "num_leaves": 2,
}


@ pytest.fixture
def tok():
    return Tokenizer(**test_params)


def test_vocab_size(tok):
    # Specials: <pad>,<bos>,<eos>
    expected = 3 + test_params["num_features"] + test_params["num_bins"] + test_params["num_leaves"]
    assert len(tok) == expected


def test_encode_feature_valid(tok):
    for i in range(test_params["num_features"]):
        tid = tok.encode_feature(i)
        assert isinstance(tid, int)
        assert tok.id2tok[tid] == f"F_{i}"


def test_encode_feature_invalid(tok):
    with pytest.raises(IndexError):
        tok.encode_feature(-1)
    with pytest.raises(IndexError):
        tok.encode_feature(test_params["num_features"])


def test_encode_threshold_valid(tok):
    for j in range(test_params["num_bins"]):
        tid = tok.encode_threshold(j)
        assert tok.id2tok[tid] == f"TH_{j}"


def test_encode_threshold_invalid(tok):
    with pytest.raises(IndexError):
        tok.encode_threshold(-1)
    with pytest.raises(IndexError):
        tok.encode_threshold(test_params["num_bins"])


def test_encode_leaf_valid(tok):
    # Two leaves: LEAF_0, LEAF_1
    tid0 = tok.encode_leaf(0)
    tid1 = tok.encode_leaf(1)
    assert tok.id2tok[tid0].startswith("LEAF")
    assert tok.id2tok[tid1].startswith("LEAF")


def test_encode_leaf_invalid(tok):
    with pytest.raises(IndexError):
        tok.encode_leaf(-1)
    with pytest.raises(IndexError):
        tok.encode_leaf(test_params["num_leaves"])


def test_decode_one(tok):
    # Feature
    tid_f = tok.encode_feature(2)
    typ_f, idx_f = tok.decode_one(tid_f)
    assert typ_f == "feature" and idx_f == 2
    # Threshold
    tid_t = tok.encode_threshold(3)
    typ_t, idx_t = tok.decode_one(tid_t)
    assert typ_t == "threshold" and idx_t == 3
    # Leaf
    tid_l = tok.encode_leaf(1)
    typ_l, idx_l = tok.decode_one(tid_l)
    assert typ_l == "leaf" and idx_l == 1


@ pytest.mark.parametrize("bad_tid", [-1, 0, 1, 2, len(SimpleTokenizer(**test_params))])
def test_decode_one_invalid(tok, bad_tid):
    with pytest.raises(ValueError):
        tok.decode_one(bad_tid)


def test_decode_sequence(tok):
    # Construct sequence: pad, bos, feature, threshold, leaf, eos
    seq = [tok.PAD, tok.BOS,
           tok.encode_feature(1), tok.encode_threshold(5), tok.encode_leaf(0),
           tok.EOS]
    decoded = tok.decode(seq)
    assert decoded == [("feature",1),("threshold",5),("leaf",0)]
