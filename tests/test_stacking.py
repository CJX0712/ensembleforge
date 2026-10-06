"""StackedGen 单元测试。作者：晨星。"""

import numpy as np

from ensembleforge.ensemble.stacking import StackedGen


def test_stacking_runs_numpy_backend():
    rng = np.random.default_rng(0)
    N, C = 300, 3
    y = rng.integers(0, C, N)
    good = np.zeros((N, C))
    good[np.arange(N), y] = 1.0
    bad = np.full((N, C), 1.0 / C)
    oof = [good, bad]
    s = StackedGen(backend="numpy", seed=1)
    s.fit_oof(oof, y)
    P = s.predict([good, bad])
    assert P.shape == (N, C)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
