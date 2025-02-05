import pytest

from sqil_core.utils._analysis import *


def test_estimate_linear_background():
    a = 5
    b = 2

    x = np.linspace(0, 10, 100)
    y = a * x + b

    [c, m] = estimate_linear_background(x, y, 1)

    assert m == pytest.approx(a)
    assert c == pytest.approx(b)
