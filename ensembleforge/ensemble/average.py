"""朴素平均集成（基线）：等权平均所有基模型概率。作者：晨星。"""

import numpy as np

from ..core.errors import EnsembleError


class NaiveAverage:
    """等权平均集成 —— 衡量「选型是否真比朴素袋装更有价值」的对照基线。"""

    name = "NaiveAvg"

    def fit_oof(self, oof_preds, y):
        if not oof_preds:
            raise EnsembleError("oof_preds 为空")
        self.n_models_ = len(oof_preds)
        self.C_ = np.asarray(oof_preds[0], dtype=float).shape[1]
        return self

    def predict(self, test_preds_all):
        stack = np.stack([np.asarray(p, dtype=float) for p in test_preds_all], axis=0)
        return stack.mean(axis=0)

    @property
    def members(self):
        return {"all": self.n_models_}
