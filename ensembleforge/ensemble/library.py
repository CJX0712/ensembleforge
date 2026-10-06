"""模型库构建：Tier-0 sklearn 库 + Tier-1 纯 numpy 离线兜底库。作者：晨星。"""

from .base_models import (
    NumpyKNN,
    NumpyLogisticRegression,
    NumpyMLP,
    SklearnWrapper,
)

__all__ = ["available_sklearn", "build_numpy_library", "build_sklearn_library"]


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


def build_sklearn_library(seed: int = 42):
    """多样化 sklearn 模型库（线性 / 袋装树 / 提升树 / 实例 / 朴素贝叶斯 / 神经网络）。

    多样性是集成选型的前提：线性、树、近邻、朴素贝叶斯、神经网络各有偏差，
    互补增益才能让 EnsembleFuse 在「非劣于最佳单模型」的同时显著优于朴素平均。
    多分类统一走 solver='lbfgs'（sklearn 1.9 已移除 multi_class 参数，liblinear 不支持多类）。
    """
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    return [
        SklearnWrapper(LogisticRegression(max_iter=2000, solver="lbfgs"), "LR"),
        SklearnWrapper(RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed), "RF"),
        SklearnWrapper(ExtraTreesClassifier(n_estimators=100, n_jobs=1, random_state=seed), "ET"),
        SklearnWrapper(GradientBoostingClassifier(n_estimators=100, random_state=seed), "GB"),
        SklearnWrapper(KNeighborsClassifier(n_neighbors=7), "KNN"),
        SklearnWrapper(GaussianNB(), "NB"),
        SklearnWrapper(
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=seed), "MLP"
        ),
        SklearnWrapper(
            MLPClassifier(hidden_layer_sizes=(16,), max_iter=100, random_state=seed + 1), "MLP2"
        ),
    ]


def build_numpy_library(seed: int = 42):
    """纯 numpy 离线兜底库（无 sklearn 时仍可用）。"""
    return [
        NumpyLogisticRegression(lr=0.1, epochs=400, l2=1e-3, seed=seed),
        NumpyKNN(k=7),
        NumpyMLP(hidden=32, lr=0.05, epochs=400, l2=1e-3, seed=seed + 1),
    ]
