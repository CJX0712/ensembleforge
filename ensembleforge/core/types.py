"""数据类型定义。作者：晨星。"""

from dataclasses import dataclass, field


@dataclass
class BenchmarkRow:
    """单条（数据集 × 数据种子）基准记录。"""

    dataset: str
    seed: int
    n_train: int
    n_test: int
    n_classes: int
    random_baseline_acc: float
    best_single_name: str
    best_single_acc: float
    best_single_logloss: float
    naive_avg_acc: float
    naive_avg_logloss: float
    fuse_acc: float
    fuse_logloss: float
    stack_acc: float
    stack_logloss: float
    base_models: dict[str, float] = field(default_factory=dict)


@dataclass
class Aggregate:
    """聚合指标（跨数据集 × 种子，mean±std + 显著性）。"""

    metric: str
    fuse_mean: float
    fuse_std: float
    best_single_mean: float
    best_single_std: float
    naive_avg_mean: float
    naive_avg_std: float
    n: int
    delta_vs_best: float
    delta_vs_avg: float
    noninferior_vs_best: bool  # G1：集成不显著弱于最强单模型（非劣）
    significant_vs_avg: bool  # G3：集成显著优于朴素平均（真实可证胜点）
