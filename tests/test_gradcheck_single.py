from micrograd import Value
from tests.gradcheck import central_diff, assert_close


def test_gradcheck_single_var():
    # define a pure float function that mirrors your Value expression
    def f_float(x: float) -> float:
        a = Value(x)
        out = (a * a + 3) * a.tanh() + a.exp()
        return out.data

    # autograd
    a = Value(1.234)
    out = (a * a + 3) * a.tanh() + a.exp()
    out.backward()
    grad_auto = a.grad

    # numerical
    grad_num = central_diff(f_float, 1.234, h=1e-6)

    assert_close(grad_auto, grad_num, tol=1e-4, msg="single-var gradcheck")
