from micrograd import Value

def test_add_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0

def test_mul_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    c.backward()
    assert a.grad == -3.0
    assert b.grad == 2.0

def test_tanh_backward():
    x = Value(0.0)
    y = x.tanh()
    y.backward()
    assert abs(x.grad - 1.0) < 1e-12
