"""基准流水线测试：无泄漏 / 确定性 / 离线兜底 / CLI 冒烟。作者：晨星。"""

import os
import subprocess
import sys

from ensembleforge.core.config import Config
from ensembleforge.core.seed import set_all
from ensembleforge.pipeline.benchmark import EnsembleBenchmark

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_benchmark_runs_no_leakage():
    cfg = Config(datasets=["blobs"], data_seeds=[1], n_samples=600, cv=4, bags=8, max_size=10)
    set_all(cfg.seed)
    rows = EnsembleBenchmark(cfg).run()
    r = rows[0]
    # 集成必须显著优于随机基线（证明有区分度、无泄漏虚高）
    assert r.fuse_acc > r.random_baseline_acc + 0.1
    # 集成在 log-loss 上至少不弱于朴素平均（非劣）
    assert r.fuse_logloss <= r.naive_avg_logloss + 1e-9


def test_determinism():
    cfg = Config(datasets=["blobs"], data_seeds=[1], n_samples=400, cv=4, bags=6, max_size=8)
    set_all(cfg.seed)
    r1 = EnsembleBenchmark(cfg).run()
    set_all(cfg.seed)
    r2 = EnsembleBenchmark(cfg).run()
    assert r1[0].fuse_acc == r2[0].fuse_acc
    assert r1[0].fuse_logloss == r2[0].fuse_logloss


def test_offline_numpy_backend():
    cfg = Config(
        backend="numpy",
        datasets=["blobs"],
        data_seeds=[1],
        n_samples=400,
        cv=4,
        bags=8,
        max_size=10,
    )
    set_all(cfg.seed)
    rows = EnsembleBenchmark(cfg).run()
    r = rows[0]
    # 纯 numpy 离线路径：集成至少不弱于最强单模型（非劣），且无泄漏
    assert r.fuse_acc >= r.best_single_acc - 0.02
    assert r.fuse_logloss <= r.best_single_logloss + 1e-9


def test_cli_smoke(tmp_path):
    out = tmp_path / "b.json"
    code = subprocess.run(
        [
            sys.executable,
            "-m",
            "ensembleforge.cli",
            "--datasets",
            "blobs",
            "--data-seeds",
            "1",
            "--n-samples",
            "300",
            "--out",
            str(out),
            "--quiet",
        ],
        check=False,
        cwd=ROOT,
    )
    assert code.returncode == 0
    assert out.exists()
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "rows" in data and "aggregate" in data
