# Changelog

## v0.1.0 (2026-10-07)
- 初版发布。
- 旗舰 `EnsembleFuse`：Caruana Bagged Ensemble Selection（B 次 bootstrap 贪婪前向选择 + 早停）。
- 二级对照 `StackedGen`：Wolpert Stacked Generalization（元学习器在 OOF 概率上训练）。
- Tier-0 模型库（sklearn，8 个）：LR / RF / ET / GB / KNN / NB / MLP / MLP2，多样性覆盖线性、袋装树、提升树、近邻、朴素贝叶斯、神经网络。
- Tier-1 纯 numpy 离线兜底库（无 sklearn 时仍可用）：numpy-LR / numpy-KNN / numpy-MLP。
- 基准聚合采用**配对差显著性**（fuse/best/avg 同任务共评测），替代直接跨数据集池化，避免类别数/尺度差异高估方差、掩盖真实胜点。
- 合成多类 DGP（blobs / moons / circles / xor）+ 标签噪声注入，难度甜点可复现。
- OOF 无泄漏基准流水线（StandardScaler 仅 fit 训练折、集成选型只用 OOF+训练标签）。
- 全局确定性（固定 seed，同 seed 两次运行核心指标逐位一致）。
- 作者：晨星 (CJX0712)。
