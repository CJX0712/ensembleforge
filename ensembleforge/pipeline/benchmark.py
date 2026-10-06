"""基准流水线：OOF 无泄漏选型 + 测试集评测 + 聚合统计。作者：晨星。

泄漏控制（pitfalls §B）：
- StandardScaler 仅在训练折上 fit。
- 基模型 OOF 概率由「分层 K 折交叉验证」得到，每样本的预测来自未见其本身的模型。
- 集成选型（EnsembleFuse / StackedGen）仅用 OOF + 训练标签；测试标签从不参与。
- 最终测试预测由各基模型在全训练集上重训后给出。
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ..core.config import Config
from ..core.types import Aggregate, BenchmarkRow
from ..data.synthetic import generate_dataset
from ..ensemble.average import NaiveAverage
from ..ensemble.ensemble_selection import EnsembleFuse
from ..ensemble.library import build_numpy_library, build_sklearn_library
from ..ensemble.stacking import StackedGen


def _accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _logloss(P, y, eps: float = 1e-12) -> float:
    P = np.clip(np.asarray(P, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y).astype(int)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def _oof_predict(model, X, y, cv) -> np.ndarray:
    """手动 OOF：对每折 fit 基模型并预测验证折（兼容 sklearn 包装与纯 numpy 模型）。"""
    n_classes = len(np.unique(y))
    out = np.zeros((len(y), n_classes))
    for tr_idx, va_idx in cv.split(X, y):
        model.fit(X[tr_idx], y[tr_idx])
        out[va_idx] = model.predict_proba(X[va_idx])
    return out


class EnsembleBenchmark:
    def __init__(self, config: Config):
        self.config = config

    def _build_library(self):
        if self.config.backend == "numpy":
            return build_numpy_library(self.config.seed)
        return build_sklearn_library(self.config.seed)

    def _run_one(self, ds_name: str, seed: int) -> BenchmarkRow:
        cfg = self.config
        X, y = generate_dataset(ds_name, n=cfg.n_samples, seed=seed, noise=cfg.noise)
        C = len(np.unique(y))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        models = self._build_library()
        cv = StratifiedKFold(n_splits=cfg.cv, shuffle=True, random_state=seed)
        oof = []
        test_preds = []
        for m in models:
            op = _oof_predict(m, Xtr_s, ytr, cv)
            m.fit(Xtr_s, ytr)
            tp = m.predict_proba(Xte_s)
            oof.append(op)
            test_preds.append(tp)

        base_acc = {
            m.name: _accuracy(yte, np.argmax(tp, axis=1)) for m, tp in zip(models, test_preds)
        }
        oof_ll = [_logloss(o, ytr) for o in oof]
        best_idx = int(np.argmin(oof_ll))
        best_name = models[best_idx].name
        best_single_acc = base_acc[best_name]
        best_single_ll = _logloss(test_preds[best_idx], yte)

        navg = NaiveAverage()
        navg.fit_oof(oof, ytr)
        navg_pred = navg.predict(test_preds)
        naive_avg_acc = _accuracy(yte, np.argmax(navg_pred, axis=1))
        naive_avg_ll = _logloss(navg_pred, yte)

        fuse = EnsembleFuse(bags=cfg.bags, max_size=cfg.max_size, patience=cfg.patience, seed=seed)
        fuse.fit_oof(oof, ytr)
        fuse_pred = fuse.predict(test_preds)
        fuse_acc = _accuracy(yte, np.argmax(fuse_pred, axis=1))
        fuse_ll = _logloss(fuse_pred, yte)

        stack = StackedGen(backend=cfg.backend, seed=seed)
        stack.fit_oof(oof, ytr)
        stack_pred = stack.predict(test_preds)
        stack_acc = _accuracy(yte, np.argmax(stack_pred, axis=1))
        stack_ll = _logloss(stack_pred, yte)

        return BenchmarkRow(
            dataset=ds_name,
            seed=seed,
            n_train=len(ytr),
            n_test=len(yte),
            n_classes=C,
            random_baseline_acc=1.0 / C,
            best_single_name=best_name,
            best_single_acc=best_single_acc,
            best_single_logloss=best_single_ll,
            naive_avg_acc=naive_avg_acc,
            naive_avg_logloss=naive_avg_ll,
            fuse_acc=fuse_acc,
            fuse_logloss=fuse_ll,
            stack_acc=stack_acc,
            stack_logloss=stack_ll,
            base_models=base_acc,
        )

    def run(self) -> list:
        rows = []
        for ds in self.config.datasets:
            for seed in self.config.data_seeds:
                rows.append(self._run_one(ds, seed))
        return rows

    @staticmethod
    def _paired_agg(metric, rows, fuse_key, best_key, avg_key) -> Aggregate:
        """配对聚合：fuse / best / avg 均在相同 (数据集×种子) 上评测，
        故用「逐行配对差」做显著性——这才是对共线任务的正确统计处理
        （直接池化跨数据集会引入类别数/尺度差异，严重高估方差、掩盖真实胜点）。

        显著性沿用 SOP 约定「均值差 > ½·σ」的配对形式：
        σ 取配对差的标准差（ddof=1）。
        - G1 非劣：ΔvsBest >= -½·σ_diff（不显著更差）
        - G3 显著优于朴素平均：ΔvsAvg  >  ½·σ_diff（真实可证胜点）
        """
        fuse = np.array([getattr(r, fuse_key) for r in rows], dtype=float)
        best = np.array([getattr(r, best_key) for r in rows], dtype=float)
        avg = np.array([getattr(r, avg_key) for r in rows], dtype=float)
        fm = float(fuse.mean())
        bm = float(best.mean())
        am = float(avg.mean())
        fs = float(fuse.std(ddof=1)) if len(fuse) > 1 else 0.0
        bs = float(best.std(ddof=1)) if len(best) > 1 else 0.0
        ast = float(avg.std(ddof=1)) if len(avg) > 1 else 0.0
        # 配对差：正值 = fuse 更优
        if metric == "logloss":
            d_best = best - fuse  # log-loss 越低越好
            d_avg = avg - fuse
        else:
            d_best = fuse - best  # accuracy 越高越好
            d_avg = fuse - avg
        db_m = float(d_best.mean())
        db_s = float(d_best.std(ddof=1)) if len(d_best) > 1 else 0.0
        da_m = float(d_avg.mean())
        da_s = float(d_avg.std(ddof=1)) if len(d_avg) > 1 else 0.0
        noninferior_vs_best = db_m >= -0.5 * db_s
        significant_vs_avg = da_m > 0.5 * da_s
        return Aggregate(
            metric=metric,
            fuse_mean=fm,
            fuse_std=fs,
            best_single_mean=bm,
            best_single_std=bs,
            naive_avg_mean=am,
            naive_avg_std=ast,
            n=len(rows),
            delta_vs_best=db_m,
            delta_vs_avg=da_m,
            noninferior_vs_best=bool(noninferior_vs_best),
            significant_vs_avg=bool(significant_vs_avg),
        )

    def aggregate_all(self, rows) -> dict:
        return {
            "accuracy": self._paired_agg(
                "accuracy", rows, "fuse_acc", "best_single_acc", "naive_avg_acc"
            ),
            "logloss": self._paired_agg(
                "logloss", rows, "fuse_logloss", "best_single_logloss", "naive_avg_logloss"
            ),
        }
