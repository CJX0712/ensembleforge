"""NaiveAverage 单元测试。作者：晨星。"""

import numpy as np

from ensembleforge.ensemble.average import NaiveAverage


def test_naive_average_equals_mean():
    rng = np.random.default_rng(1)
    a = rng.random((10, 3))
    b = rng.random((10, 3))
    c = rng.random((10, 3))
    navg = NaiveAverage()
    navg.fit_oof([a, b, c], np.zeros(10, dtype=int))
    P = navg.predict([a, b, c])
    assert np.allclose(P, (a + b + c) / 3)
