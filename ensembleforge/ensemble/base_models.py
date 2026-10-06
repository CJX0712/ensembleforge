"""基础模型：sklearn 包装 + 纯 numpy 离线兜底。作者：晨星。

所有模型统一实现 BaseModelProto：name / fit / predict / predict_proba，
predict_proba 返回 [N, C] 且每行和为 1、列序对应标签 0..C-1。
"""

import numpy as np

__all__ = [
    "NumpyKNN",
    "NumpyLogisticRegression",
    "NumpyMLP",
    "SklearnWrapper",
]


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


class SklearnWrapper:
    """把任意 sklearn  estimator（已配好超参/随机种子）包成统一接口。"""

    def __init__(self, estimator, name: str):
        self.estimator = estimator
        self.name = name

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


class NumpyLogisticRegression:
    """纯 numpy 多分类 Logistic Regression（softmax + 交叉熵 + L2，全批梯度下降）。"""

    name = "numpy-LR"

    def __init__(self, lr: float = 0.1, epochs: int = 400, l2: float = 1e-3, seed: int = 0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_
        col = np.searchsorted(self.classes_, y)
        Y = np.zeros((len(y), C))
        Y[np.arange(len(y)), col] = 1.0
        rng = np.random.default_rng(self.seed)
        W = rng.normal(0.0, 0.01, (Xs.shape[1], C))
        b = np.zeros(C)
        for _ in range(self.epochs):
            p = _softmax(Xs @ W + b)
            gW = (Xs.T @ (p - Y)) / len(y) + self.l2 * W
            gb = (p - Y).mean(axis=0)
            W -= self.lr * gW
            b -= self.lr * gb
        self.W_ = W
        self.b_ = b
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        return _softmax(Xs @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class NumpyKNN:
    """纯 numpy k-近邻（标准化欧氏距离，投票概率）。"""

    name = "numpy-KNN"

    def __init__(self, k: int = 7):
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        mean = self.X_.mean(axis=0)
        std = self.X_.std(axis=0) + 1e-8
        Xs = (X - mean) / std
        Ts = (self.X_ - mean) / std
        d = np.sqrt(((Xs[:, None, :] - Ts[None, :, :]) ** 2).sum(axis=-1))
        idx = np.argsort(d, axis=1)[:, : self.k]
        nq = X.shape[0]
        C = len(self.classes_)
        P = np.zeros((nq, C))
        for i in range(nq):
            P[i] = np.bincount(self.y_[idx[i]], minlength=C) / self.k
        return P

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class NumpyMLP:
    """纯 numpy 单隐层 MLP（tanh 隐层 + softmax 输出，SGD + L2）。"""

    name = "numpy-MLP"

    def __init__(
        self, hidden: int = 32, lr: float = 0.05, epochs: int = 400, l2: float = 1e-3, seed: int = 0
    ):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_
        col = np.searchsorted(self.classes_, y)
        Y = np.zeros((len(y), C))
        Y[np.arange(len(y)), col] = 1.0
        rng = np.random.default_rng(self.seed)
        d0 = Xs.shape[1]
        W1 = rng.normal(0.0, np.sqrt(2.0 / d0), (d0, self.hidden))
        b1 = np.zeros(self.hidden)
        W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden), (self.hidden, C))
        b2 = np.zeros(C)
        for _ in range(self.epochs):
            h = np.tanh(Xs @ W1 + b1)
            p = _softmax(h @ W2 + b2)
            dl = (p - Y) / len(y)
            gW2 = h.T @ dl + self.l2 * W2
            gb2 = dl.sum(axis=0)
            dh = dl @ W2.T * (1.0 - h**2)
            gW1 = Xs.T @ dh + self.l2 * W1
            gb1 = dh.sum(axis=0)
            W2 -= self.lr * gW2
            b2 -= self.lr * gb2
            W1 -= self.lr * gW1
            b1 -= self.lr * gb1
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        h = np.tanh(Xs @ self.W1 + self.b1)
        return _softmax(h @ self.W2 + self.b2)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
