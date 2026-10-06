"""StackedGen：Wolpert Stacked Generalization（二级元学习器）。作者：晨星。

- 元学习器在基模型的 OOF 概率（拼接成 [N, C*M] 特征）上训练。
- 测试时用基模型在全训练集上重训后的概率喂元学习器得到集成预测。
- 元学习器默认 sklearn LogisticRegression（后端=sklearn），否则用纯 numpy LR 兜底。
"""

import numpy as np

from ..core.errors import EnsembleError
from .base_models import NumpyLogisticRegression


class StackedGen:
    name = "StackedGen"

    def __init__(self, backend: str = "sklearn", seed: int = 42):
        self.backend = backend
        self.seed = seed
        self.meta_ = None
        self.n_models_ = 0
        self.C_ = 0

    def fit_oof(self, oof_preds, y):
        if not oof_preds:
            raise EnsembleError("oof_preds 为空")
        Z = np.concatenate([np.asarray(p, dtype=float) for p in oof_preds], axis=1)
        y = np.asarray(y).astype(int)
        if self.backend == "sklearn":
            from sklearn.linear_model import LogisticRegression

            self.meta_ = LogisticRegression(max_iter=2000, solver="lbfgs")
        else:
            self.meta_ = NumpyLogisticRegression(lr=0.1, epochs=400, l2=1e-3, seed=self.seed)
        self.meta_.fit(Z, y)
        self.n_models_ = len(oof_preds)
        self.C_ = np.asarray(oof_preds[0], dtype=float).shape[1]
        return self

    def predict(self, test_preds_all):
        if self.meta_ is None:
            raise EnsembleError("尚未 fit_oof，无法 predict")
        Z = np.concatenate([np.asarray(p, dtype=float) for p in test_preds_all], axis=1)
        return self.meta_.predict_proba(Z)

    @property
    def members(self):
        return {"meta_inputs": self.n_models_}
