"""EnsembleFuse（Bagged ES）单元测试。作者：晨星。"""

import numpy as np

from ensembleforge.ensemble.ensemble_selection import EnsembleFuse


def _make():
    rng = np.random.default_rng(0)
    N, C = 300, 3
    y = rng.integers(0, C, N)
    good = np.zeros((N, C))
    good[np.arange(N), y] = 1.0
    bad = np.full((N, C), 1.0 / C)
    return [good, bad, bad], y


def test_selects_dominant():
    oof, y = _make()
    fuse = EnsembleFuse(bags=8, max_size=8, patience=2, seed=1)
    fuse.fit_oof(oof, y)
    assert fuse.weights_.get(0, 0) >= fuse.weights_.get(1, 0)


def test_deterministic():
    oof, y = _make()
    f1 = EnsembleFuse(seed=5)
    f1.fit_oof(oof, y)
    f2 = EnsembleFuse(seed=5)
    f2.fit_oof(oof, y)
    assert f1.weights_ == f2.weights_


def test_predict_equals_best_when_dominant():
    oof, y = _make()
    fuse = EnsembleFuse(bags=8, max_size=8, patience=2, seed=1)
    fuse.fit_oof(oof, y)
    test = [oof[0], oof[1], oof[1]]  # 模型0 应主导
    P = fuse.predict(test)
    assert P.shape == oof[0].shape
    assert np.allclose(P, oof[0])
