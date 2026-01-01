from micrograd import Value
from tests.gradcheck import central_diff, assert_close



def test_gradcheck_three_vars():
    # float version of the same function
    def f_float(a: float, b: float, c: float) -> float:
        A, B, C = Value(a), Value(b), Value(c)
        out = -A**3 + (B * 3).tanh() + (B**2.5) - (A**0.5) - (1.0 / C)
        return out.data

    # Value graph version
    a0, b0, c0 = 2.0, 3.0, 4.0
    A, B, C = Value(a0), Value(b0), Value(c0)
    out = -A**3 + (B * 3).tanh() + (B**2.5) - (A**0.5) - (1.0 / C)
    out.backward()

    grads_auto = [A.grad, B.grad, C.grad]

    # numerical gradients: perturb one variable at a time
    def fa(x): return f_float(x, b0, c0)
    def fb(x): return f_float(a0, x, c0)
    def fc(x): return f_float(a0, b0, x)

    grads_num = [
        central_diff(fa, a0, h=1e-6),
        central_diff(fb, b0, h=1e-6),
        central_diff(fc, c0, h=1e-6),
    ]

    for i, (ga, gn) in enumerate(zip(grads_auto, grads_num)):
        assert_close(ga, gn, tol=1e-4, msg=f"gradcheck dim {i}")
