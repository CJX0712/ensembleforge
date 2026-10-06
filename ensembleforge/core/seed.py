"""全局确定性入口。作者：晨星。"""

import random

import numpy as np

__all__ = ["make_rng", "set_all"]


def set_all(seed: int) -> int:
    """一次性设定全局随机种子（legacy numpy + python random），保证可复现。

    返回实际生效的 seed，便于审计。所有模块的随机数流都从这一入口派生。
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def make_rng(seed):
    """返回独立、可复现的 numpy Generator（优先于 legacy global state）。"""
    return np.random.default_rng(seed)
