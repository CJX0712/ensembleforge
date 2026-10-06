"""确定性入口测试。作者：晨星。"""

import numpy as np

from ensembleforge.core.seed import make_rng, set_all


def test_set_all_idempotent():
    set_all(7)
    a = np.random.rand(5)
    set_all(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_make_rng_deterministic():
    r1 = make_rng(3)
    r2 = make_rng(3)
    assert np.array_equal(r1.random(4), r2.random(4))
