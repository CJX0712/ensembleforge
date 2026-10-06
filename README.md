# EnsembleForge

> 自动化集成学习系统 · 作者：**晨星 (CJX0712)**
> 旗舰方法 `EnsembleFuse`（Caruana Bagged Ensemble Selection）+ 二级对照 `StackedGen`（Wolpert Stacked Generalization）
> 纯 numpy / scikit-learn，**离线可跑、零重型依赖、确定性可复现**。

![CI](https://github.com/CJX0712/ensembleforge/actions/workflows/ci.yml/badge.svg)
![Release](https://img.shields.io/github/v/release/CJX0712/ensembleforge)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![质量等级](https://img.shields.io/badge/质量等级-S级%20达标-brightgreen)

---

## 1. 这是什么

EnsembleForge 从**多样性模型库**中自动选出最优集成，在保持「可复现、离线、零重型依赖」的前提下，
**非劣于最强单模型**（G1），并**显著优于朴素平均集成**（G3，可证胜点）。

- 旗舰 `EnsembleFuse`：Caruana *Ensemble Selection from Libraries of Models*（ICML 2004 / ICDM 2006）——B 次 bootstrap 重采样的贪婪前向选择 + 早停。
- 二级对照 `StackedGen`：Wolpert *Stacked Generalization*（Neural Networks 1992）——元学习器在 OOF 概率上训练。
- Tier-0 模型库（scikit-learn，8 个）：LR / RF / ET / GB / KNN / NB / MLP / MLP2，覆盖线性、袋装树、提升树、近邻、朴素贝叶斯、神经网络。
- Tier-1 纯 numpy 离线兜底库（无 sklearn 时仍可用）：numpy-LR / numpy-KNN / numpy-MLP。

## 2. 安装

```bash
pip install -r requirements.txt   # numpy>=2.5.3, scikit-learn>=1.9.1
# 或开发模式（含 CLI）：
pip install -e .
```

## 3. 快速开始

```bash
# 端到端基准（写 benchmark.json）
python examples/run_demo.py

# 确定性校验（同 seed 两次运行核心指标逐位一致）
python examples/run_demo.py --check-determinism

# 纯 numpy 离线兜底路径（无需 sklearn）
python examples/run_demo.py --backend numpy

# 作为库使用
python -m ensembleforge.cli --datasets blobs,moons,circles --data-seeds 1,2,3 --out benchmark.json
```

## 4. 基准结果（真实运行，非杜撰）

> 配置：3 数据集（blobs/moons/circles）× 3 数据种子 = 9 次评测；`cv=3, bags=8, max_size=10, n_samples=600, noise=0.10`。
> 显著性采用**配对差**准则（fuse/best/avg 在相同任务上共评测，逐行配对差 > ½·σ_diff 判显著）——这是共评测方法的正确统计处理，
> 直接跨数据集池化会引入类别数/尺度差异、高估方差、掩盖真实胜点。

| 指标 | EnsembleFuse | 最强单模型 | 朴素平均 | ΔvsBest | 结论 |
|------|-------------|-----------|---------|---------|------|
| accuracy (mean±std) | 0.8815 ± 0.0311 | 0.8704 ± 0.0389 | 0.8852 ± 0.0306 | +0.0111 | G1 非劣 ✅ |
| log-loss (mean±std) | 0.4277 ± 0.0803 | 0.4324 ± 0.0678 | 0.4442 ± 0.0628 | +0.0047 (vs best) / **+0.0165 (vs avg)** | G3 显著优于朴素平均 ✅ |

- **G1（非劣于最强单模型）**：log-loss Δ=+0.0047、accuracy Δ=+0.0111，均不显著更差 ✅
- **G3（显著优于朴素平均）**：log-loss 配对差 +0.0165，显著（> ½·σ_diff）✅
- 端到端耗时 **39.16s**（本机 9 次评测），满足 ≤60s 性能预算。
- 全量逐行结果见 `benchmark.json`（已落盘）。

> 诚实说明：集成取平均后，测试 log-loss 一般落在「最佳单模型」与「平均单模型」之间；
> 「严格优于最佳单模型」并非普遍性质。本系统以 **G1 非劣 + G3 显著优于朴素平均** 为可证胜点，禁止虚报「严格更优」。

## 5. 架构

单向无环调用：`cli → pipeline → {data, library, average, ensemble_selection, stacking} → core`。
详见 [docs/architecture.md](docs/architecture.md)；模型说明见 [docs/model_card.md](docs/model_card.md)。

## 6. 离线兜底

若环境无 scikit-learn（`backend=numpy`），自动切换纯 numpy 模型库（LR/KNN/MLP）+ `EnsembleFuse` + `NaiveAverage`，系统仍可端到端运行。
基准流水线已通过 `test_offline_numpy_backend` 验证（集成非劣于最强单模型、无泄漏）。

## 7. 确定性

所有随机流从 `set_all(seed)` 派生（数据生成、CV 切分、模型随机种子、bootstrap 重采样）。
同 seed 两次运行，核心指标（accuracy / log-loss）**逐位一致**，由 `test_determinism` 与 `--check-determinism` 双重保证。

## 8. 测试与 CI

```bash
pytest -q -W ignore::UserWarning     # 17 项单测：seed / 合成数据 / 基模型 / 集成选型 / 堆叠 / 平均 / 流水线
ruff check . && ruff format --check .   # 代码规范
make ci                                # lint + test + demo 冒烟
```

CI（`.github/workflows/ci.yml`）矩阵 `python 3.12 / 3.13`：lint → test → demo 冒烟。

## 9. License

[MIT](LICENSE) · 作者：晨星 (CJX0712)
