# EnsembleForge 架构

> 作者：晨星 (CJX0712) · 自动化集成学习系统

## 1. 设计目标

从**多样性模型库**中自动选出最优集成，在保持可复现、离线、零重型依赖的前提下，
可复现地超越「最强单模型」与「朴素平均集成」。

## 2. 固定骨架（单向无环）

```
cli.py
  └─> pipeline/benchmark.py  (EnsembleBenchmark.run / aggregate_all)
        ├─> data/synthetic.py        (合成多类 DGP)
        ├─> ensemble/library.py      (模型库构建)
        ├─> ensemble/average.py      (NaiveAverage 基线)
        ├─> ensemble/ensemble_selection.py (EnsembleFuse 旗舰)
        └─> ensemble/stacking.py     (StackedGen 二级对照)
              └─> core/ (config / seed / types / errors / interfaces)
```

调用方向严格单向：`cli → pipeline → {data, library, average, ensemble_selection, stacking} → core`。
各模块可独立验证（单测 + 最小可运行示例）。

## 3. 模块职责

| 模块 | 职责 |
|------|------|
| `core.seed` | 全局确定性入口 `set_all(seed)` / `make_rng(seed)` |
| `core.config` | `ENV_ENSEMBLE_*` 覆盖 + schema 校验 |
| `core.types` | `BenchmarkRow` / `Aggregate` 数据结构 |
| `data.synthetic` | `generate_dataset`：blobs / moons / circles / xor + 标签噪声 |
| `ensemble.library` | `build_sklearn_library`（Tier-0）/ `build_numpy_library`（Tier-1 兜底） |
| `ensemble.ensemble_selection` | `EnsembleFuse`：Caruana Bagged Ensemble Selection |
| `ensemble.stacking` | `StackedGen`：Wolpert Stacked Generalization |
| `ensemble.average` | `NaiveAverage`：等权平均基线 |
| `pipeline.benchmark` | OOF 无泄漏选型 + 测试集评测 + 聚合统计 |

## 4. 核心算法：EnsembleFuse（Bagged ES）

1. 对每个基模型做分层 K 折交叉验证，得到 OOF 概率预测（每样本由其未见自身的模型预测）。
2. 以 OOF + 训练标签为选型信号，做 **B 次 bootstrap 重采样**的「贪婪前向选择」：
   - 每步把令（重采样上）log-loss 降幅最大的候选模型以等权并入集成。
   - 连续 `patience` 步无改善即早停，防止过拟合与集成膨胀。
3. 跨 B 个袋收集所有被选中模型，最终集成 = 等权平均，权重正比于被选中频次。
4. 测试预测：各基模型在全训练集上重训后，按上述权重组合。

性质：单模型本身在候选集内 ⇒ 集成在选型准则上**至少不弱于最佳单模型**；
通过方差抵消与互补增益，在测试 log-loss 上通常**非劣于最佳单模型、且显著优于朴素平均**。
（注：集成取平均后，测试 log-loss 一般落在「最佳单模型」与「平均单模型」之间；
「严格优于最佳单模型」并非普遍性质，诚实结论为「非劣 + 显著优于朴素平均」。）

## 5. 泄漏控制（pitfalls §B）

- `StandardScaler` 仅在训练折上 `fit`。
- OOF 概率由 CV 得到，测试标签从不参与选型。
- 集成最终在独立测试集上评测。

## 6. 确定性

所有随机流从 `set_all(seed)` 派生：数据生成、CV 切分、模型随机种子、bootstrap 重采样。
同 seed 两次运行，核心指标（accuracy / log-loss）**逐位一致**。

## 7. 离线兜底

若 sklearn 不可用（`backend=numpy`），自动切换到纯 numpy 模型库（LR/KNN/MLP）+
`EnsembleFuse`（不依赖 sklearn）+ `NaiveAverage`，系统仍可端到端运行。

## 8. SOTA 对标

| 方法 | 来源 |
|------|------|
| Ensemble Selection from Libraries of Models | Caruana et al., ICML 2004 / ICDM 2006 |
| Stacked Generalization | Wolpert, Neural Networks 1992 |
| 基模型库 | scikit-learn（LR / RF / ET / GB / KNN / NB / MLP / MLP2，共 8 个）|
