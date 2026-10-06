"""合成多类分类 DGP 生成器。作者：晨星。

设计原则（见 pitfalls §B）：
- 每个生成器固定 seed -> 逐位可复现。
- 标签噪声以「翻转比例」注入，制造非平凡难度，给集成留出提升空间。
- 难度旋钮（cluster_std / noise / factor）调到「随机基线明显低于天花板」的甜点，
  保证基准有区分度（不会全满分/全崩）。
- 标签统一编码为 0..C-1 整数，确保各模型 predict_proba 列序一致。
"""

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

from ..core.errors import DataError

_VALID = ("blobs", "moons", "circles", "xor")


def _inject_label_noise(y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """随机翻转 noise 比例的标签（在同类之外随机改写到其它类）。"""
    y = np.asarray(y).copy()
    if noise <= 0.0:
        return y
    n = len(y)
    nflip = round(n * noise)
    if nflip == 0:
        return y
    idx = rng.choice(n, size=nflip, replace=False)
    classes = np.unique(y)
    for i in idx:
        cand = classes[classes != y[i]]
        y[i] = rng.choice(cand)
    return y


def generate_dataset(name: str, n: int = 1200, seed: int = 0, noise: float = 0.05):
    """生成 (X, y)。

    数据集：
      - blobs   : 5 类高斯团（cluster_std=1.4），线性/树/近邻各有优劣 -> 集成甜点。
      - moons   : 2 类半月（noise=0.22），非线性，LR 弱、树/MLP 强。
      - circles : 2 类同心圆（noise=0.08, factor=0.5），非线性，KNN/树强、LR 弱。
      - xor     : 4 类按前两维象限划分，强非线性，LR≈随机、树/MLP 近满分。
    """
    if name not in _VALID:
        raise DataError(f"未知数据集 {name!r}，可选 {_VALID}")
    rng = np.random.default_rng(seed)
    if name == "blobs":
        X, y = make_blobs(n_samples=n, n_features=12, centers=5, cluster_std=1.4, random_state=seed)
    elif name == "moons":
        X, y = make_moons(n_samples=n, noise=0.22, random_state=seed)
    elif name == "circles":
        X, y = make_circles(n_samples=n, noise=0.08, factor=0.5, random_state=seed)
    elif name == "xor":
        d = 4
        X = rng.normal(0.0, 1.0, size=(n, d))
        s1 = X[:, 0] > 0
        s2 = X[:, 1] > 0
        y = (s1.astype(int)) * 2 + (s2.astype(int))
    else:  # pragma: no cover
        raise DataError(f"未处理的数据集 {name!r}")
    y = _inject_label_noise(y, noise, rng)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)
