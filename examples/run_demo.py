"""EnsembleForge 端到端演示：跑基准并落盘 benchmark.json。作者：晨星。

从仓库根执行：
  python examples/run_demo.py            # 默认配置，写 benchmark.json
  python examples/run_demo.py --check-determinism   # 额外做二次运行确定性校验
  python examples/run_demo.py --backend numpy        # 纯 numpy 离线兜底路径
"""

import argparse
import json
import os
import sys
import time
import warnings

# 静默 MLP 收敛警告（迭代上限内未收敛属正常弱模型行为，不影响结论）
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

# 允许以脚本方式直接运行（把仓库根加入 sys.path）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ensembleforge.core.config import Config
from ensembleforge.core.seed import set_all
from ensembleforge.pipeline.benchmark import EnsembleBenchmark

_OUT = os.path.join(_ROOT, "benchmark.json")


def _print(rows, agg):
    print("\n=== EnsembleForge demo ===")
    print(
        f"{'dataset':<9}{'seed':>5}{'C':>4}{'rand':>7}{'best':>7}{'avg':>7}{'fuse':>7}{'stack':>7}"
    )
    for r in rows:
        print(
            f"{r.dataset:<9}{r.seed:>5}{r.n_classes:>4}"
            f"{r.random_baseline_acc:>7.3f}{r.best_single_acc:>7.3f}"
            f"{r.naive_avg_acc:>7.3f}{r.fuse_acc:>7.3f}{r.stack_acc:>7.3f}"
        )
    a = agg["accuracy"]
    l = agg["logloss"]
    print(f"\n--- 聚合 (n={a.n}) ---")
    print(
        f"accuracy fuse={a.fuse_mean:.4f}±{a.fuse_std:.4f}  "
        f"best={a.best_single_mean:.4f}±{a.best_single_std:.4f}  "
        f"avg={a.naive_avg_mean:.4f}±{a.naive_avg_std:.4f}  "
        f"ΔvsBest={a.delta_vs_best:+.4f} G1非劣={a.noninferior_vs_best}"
    )
    print(
        f"logloss  fuse={l.fuse_mean:.4f}±{l.fuse_std:.4f}  "
        f"best={l.best_single_mean:.4f}±{l.best_single_std:.4f}  "
        f"avg={l.naive_avg_mean:.4f}±{l.naive_avg_std:.4f}  "
        f"ΔvsAvg={l.delta_vs_avg:+.4f} G3优于朴素平均={l.significant_vs_avg}"
    )


def _run_once(cfg: Config):
    set_all(cfg.seed)
    bench = EnsembleBenchmark(cfg)
    t0 = time.time()
    rows = bench.run()
    agg = bench.aggregate_all(rows)
    elapsed = time.time() - t0
    return rows, agg, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="EnsembleForge demo")
    ap.add_argument("--backend", choices=["sklearn", "numpy"], default=None)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--data-seeds", default=None)
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--out", default=_OUT)
    args = ap.parse_args()

    cfg = Config(
        backend=args.backend,
        datasets=args.datasets,
        data_seeds=args.data_seeds,
    )
    rows, agg, elapsed = _run_once(cfg)
    payload = {
        "config": cfg.as_dict(),
        "rows": [r.__dict__ for r in rows],
        "aggregate": {k: v.__dict__ for k, v in agg.items()},
        "elapsed_sec": round(elapsed, 3),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _print(rows, agg)
    print(f"\n耗时 {elapsed:.2f}s | benchmark 写入: {args.out}")

    if args.check_determinism:
        rows2, agg2, _ = _run_once(cfg)
        same = (
            abs(agg["accuracy"].fuse_mean - agg2["accuracy"].fuse_mean) < 1e-12
            and abs(agg["logloss"].fuse_mean - agg2["logloss"].fuse_mean) < 1e-12
            and all(r.fuse_acc == r2.fuse_acc for r, r2 in zip(rows, rows2))
        )
        print("\n[determinism] 同 seed 二次运行核心指标逐位一致:", "PASS" if same else "FAIL")
        if not same:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
