"""合成数据生成器测试。作者：晨星。"""

import numpy as np

from ensembleforge.data.synthetic import generate_dataset


def test_shapes_and_labels():
    for name in ["blobs", "moons", "circles", "xor"]:
        X, y = generate_dataset(name, n=400, seed=1, noise=0.05)
        assert X.shape[0] == 400
        assert X.ndim == 2
        assert y.min() >= 0 and y.max() < 10
        assert set(np.unique(y).tolist()).issubset(set(range(10)))


def test_reproducible():
    X1, y1 = generate_dataset("blobs", n=200, seed=42, noise=0.0)
    X2, y2 = generate_dataset("blobs", n=200, seed=42, noise=0.0)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_label_noise_changes_distribution():
    _, y0 = generate_dataset("blobs", n=1000, seed=1, noise=0.0)
    _, y1 = generate_dataset("blobs", n=1000, seed=1, noise=0.10)
    assert not np.array_equal(y0, y1)


def test_unknown_dataset_raises():
    import pytest

    from ensembleforge.core.errors import DataError

    with pytest.raises(DataError):
        generate_dataset("nope", n=100, seed=1)
