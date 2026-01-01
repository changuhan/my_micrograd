from micrograd import Value
from micrograd.losses import softmax, nll_loss

def test_softmax_sums_to_one():
    logits = [Value(1.0), Value(2.0), Value(3.0)]
    probs = softmax(logits)
    s = sum(p.data for p in probs)
    assert abs(s - 1.0) < 1e-10

def test_nll_known_value():
    logits = [Value(0.0), Value(0.0)]
    probs = softmax(logits)
    loss = nll_loss(probs, target_index=0)
    assert abs(loss.data - 0.6931471805599453) < 1e-10
