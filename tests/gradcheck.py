def central_diff(f, x, h =1e-6):
    """Numerical derivatiive of scalar function f at scalar x."""
    return (f(x+h) - f(x-h)) / (2 * h)

def assert_close(a, b, tol=1e-4, msg=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg} | {a=} {b=} diff={abs(a-b)} > {tol}")