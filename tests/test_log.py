from micrograd import Value
from tests.gradcheck import central_diff, assert_close

def test_log_backward_basic():
    a = Value(2.0)
    out = a.log()
    out.backward()
    assert_close(a.grad, 1/2.0, tol=1e-12, msg="d/dx log(x)")

def test_log_gradcheck():
    def f_float(x: float) -> float:
        a = Value(x)
        out = (a.exp() + 3).log()
        return out.data

    a = Value(1.7)
    out = (a.exp() + 3).log()
    out.backward()

    grad_num = central_diff(f_float, 1.7, h=1e-6)
    assert_close(a.grad, grad_num, tol=1e-4, msg="log gradcheck")
