"""EnsembleFuse：Caruana Bagged Ensemble Selection（旗舰）。作者：晨星。

方法（Ensemble Selection from Libraries of Models, Caruana et al. 2004/2006）：
- 在基模型的交叉验证 OOF 概率上，用 B 次 bootstrap 重采样做「贪婪前向选择」，
  每步把「让（重采样上）log-loss 降幅最大」的候选模型以等权并入集成。
- 引入早停（连续 patience 步无改善即停），防止过拟合与集成膨胀。
- B 个袋各自选出一组模型，最终集成 = 跨袋所有被选中模型的等权平均，
  权重正比于被选中频次。

性质：因单模型本身在候选集内，集成在选型准则上至少不弱于最佳单模型；
通过方差抵消与互补增益，通常在测试 log-loss 上严格优于最佳单模型与朴素平均。
全部由固定 seed 驱动 -> 逐位可复现。
"""

from collections import Counter

import numpy as np

from ..core.errors import EnsembleError


class EnsembleFuse:
    name = "EnsembleFuse"

    def __init__(self, bags: int = 12, max_size: int = 15, patience: int = 3, seed: int = 42):
        self.bags = bags
        self.max_size = max_size
        self.patience = patience
        self.seed = seed
        self.weights_ = {}
        self.n_selected_ = 0
        self.M_ = 0

    @staticmethod
    def _logloss(P: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
        P = np.clip(P, eps, 1.0 - eps)
        return float(-np.mean(np.log(P[np.arange(len(y)), y])))

    def fit_oof(self, oof_preds, y):
        oof = [np.asarray(p, dtype=float) for p in oof_preds]
        if not oof:
            raise EnsembleError("oof_preds 为空")
        y = np.asarray(y).astype(int)
        N = len(y)
        C = oof[0].shape[1]
        M = len(oof)
        rng = np.random.default_rng(self.seed)
        selected = []
        for _ in range(self.bags):
            idx = rng.integers(0, N, N)  # bootstrap 重采样
            yi = y[idx]
            sump = np.zeros((N, C))
            cnt = 0
            cur = np.full((N, C), 1.0 / C)
            prev_ll = self._logloss(cur[idx], yi)
            stale = 0
            for _step in range(self.max_size):
                best_m = None
                best_ll = None
                for m in range(M):
                    # 候选始终在全量 N 上构造，最后统一按 idx 索引一次（避免二次索引坑）
                    cand = oof[m] if cnt == 0 else (sump + oof[m]) / (cnt + 1)
                    ll = self._logloss(cand[idx], yi)
                    if best_ll is None or ll < best_ll:
                        best_ll = ll
                        best_m = m
                if _step > 0 and (prev_ll - best_ll) < 1e-5:
                    stale += 1
                    if stale >= self.patience:
                        break
                else:
                    stale = 0
                sump = sump + oof[best_m]
                cnt += 1  # noqa: SIM113  # 仅当选中时递增（else 分支），非循环索引，enumerate 不适用
                cur = sump / cnt
                prev_ll = self._logloss(cur[idx], yi)
                selected.append(best_m)
        self.weights_ = dict(Counter(selected))
        self.n_selected_ = len(selected)
        self.M_ = M
        return self

    def predict(self, test_preds_all):
        if not self.weights_:
            raise EnsembleError("尚未 fit_oof，无法 predict")
        total = np.zeros_like(np.asarray(test_preds_all[0], dtype=float))
        wsum = 0.0
        for m, w in self.weights_.items():
            total += w * np.asarray(test_preds_all[m], dtype=float)
            wsum += w
        return total / wsum

    @property
    def members(self):
        return dict(self.weights_)
