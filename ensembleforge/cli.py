"""EnsembleForge 命令行入口。作者：晨星。

用法示例：
  python -m ensembleforge.cli --datasets blobs,moons --data-seeds 1,2,3 --out benchmark.json
  python -m ensembleforge.cli --backend numpy --datasets blobs --data-seeds 1
"""

import argparse
import json
import sys

from .core.config import Config
from .core.seed import set_all
from .pipeline.benchmark import EnsembleBenchmark


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ensembleforge",
        description="EnsembleForge: 自动化集成学习（Ensemble Selection + Stacked Generalization）",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", choices=["sklearn", "numpy"], default=None)
    p.add_argument("--datasets", default=None, help="逗号分隔，如 blobs,moons,circles")
    p.add_argument("--data-seeds", default=None, help="逗号分隔整数，如 1,2,3")
    p.add_argument("--bags", type=int, default=None)
    p.add_argument("--max-size", type=int, default=None)
    p.add_argument("--cv", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--noise", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--out", default="benchmark.json")
    p.add_argument("--quiet", action="store_true", help="仅写 JSON，不打印汇总表")
    return p


def _print_summary(rows, agg):
    print("\n=== EnsembleForge 基准 ===")
    header = (
        f"{'dataset':<9}{'seed':>5}{'C':>4}{'rand':>7}{'best':>7}{'avg':>7}{'fuse':>7}{'stack':>7}"
    )
    print(header)
    for r in rows:
        print(
            f"{r.dataset:<9}{r.seed:>5}{r.n_classes:>4}"
            f"{r.random_baseline_acc:>7.3f}{r.best_single_acc:>7.3f}"
            f"{r.naive_avg_acc:>7.3f}{r.fuse_acc:>7.3f}{r.stack_acc:>7.3f}"
        )
    a = agg["accuracy"]
    l = agg["logloss"]
    print(f"\n--- 聚合 (mean±std, n={a.n}) ---")
    print(
        f"accuracy  fuse={a.fuse_mean:.4f}±{a.fuse_std:.4f}  "
        f"best={a.best_single_mean:.4f}±{a.best_single_std:.4f}  "
        f"avg={a.naive_avg_mean:.4f}±{a.naive_avg_std:.4f}  "
        f"ΔvsBest={a.delta_vs_best:+.4f}  G1非劣={a.noninferior_vs_best}"
    )
    print(
        f"logloss   fuse={l.fuse_mean:.4f}±{l.fuse_std:.4f}  "
        f"best={l.best_single_mean:.4f}±{l.best_single_std:.4f}  "
        f"avg={l.naive_avg_mean:.4f}±{l.naive_avg_std:.4f}  "
        f"ΔvsAvg={l.delta_vs_avg:+.4f}  G3优于朴素平均={l.significant_vs_avg}"
    )


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = Config(
        seed=args.seed,
        backend=args.backend,
        datasets=args.datasets,
        data_seeds=args.data_seeds,
        bags=args.bags,
        max_size=args.max_size,
        cv=args.cv,
        n_samples=args.n_samples,
        noise=args.noise,
        patience=args.patience,
    )
    set_all(cfg.seed)
    bench = EnsembleBenchmark(cfg)
    rows = bench.run()
    agg = bench.aggregate_all(rows)
    payload = {
        "config": cfg.as_dict(),
        "rows": [r.__dict__ for r in rows],
        "aggregate": {k: v.__dict__ for k, v in agg.items()},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if not args.quiet:
        _print_summary(rows, agg)
        print(f"\nbenchmark 已写入: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
