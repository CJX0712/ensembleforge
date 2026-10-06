"""基础模型测试（sklearn 库 + numpy 兜底库）。作者：晨星。"""

import numpy as np
from sklearn.model_selection import train_test_split

from ensembleforge.data.synthetic import generate_dataset
from ensembleforge.ensemble.library import build_numpy_library, build_sklearn_library


def _split(name="blobs"):
    X, y = generate_dataset(name, n=400, seed=1, noise=0.0)
    return train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


def test_proba_valid_sklearn():
    Xtr, Xte, ytr, _ = _split()
    for m in build_sklearn_library(1):
        m.fit(Xtr, ytr)
        P = m.predict_proba(Xte)
        assert P.shape == (len(Xte), len(np.unique(ytr)))
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert (P >= -1e-9).all()
        assert m.predict(Xte).shape[0] == len(Xte)


def test_proba_valid_numpy():
    Xtr, Xte, ytr, _ = _split()
    for m in build_numpy_library(1):
        m.fit(Xtr, ytr)
        P = m.predict_proba(Xte)
        assert P.shape == (len(Xte), len(np.unique(ytr)))
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert (P >= -1e-9).all()
