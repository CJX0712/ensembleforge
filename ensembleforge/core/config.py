"""配置与 ENV_* 覆盖 + schema 校验。作者：晨星。"""

import os

from .errors import ConfigError

_ENV_PREFIX = "ENV_ENSEMBLE_"


def _cast_int(v):
    try:
        return int(v)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"无法将 {v!r} 解析为整数") from exc


def _cast_float(v):
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"无法将 {v!r} 解析为浮点数") from exc


def _cast_int_list(v):
    return [_cast_int(x) for x in str(v).split(",") if x.strip() != ""]


def _cast_str_list(v):
    return [x.strip() for x in str(v).split(",") if x.strip() != ""]


class Config:
    """运行时配置。

    优先级：显式 override（CLI/代码）> 环境变量 ENV_ENSEMBLE_* > 内置默认值。
    构造后调用 validate() 做 schema 校验（正整数、后端合法等）。
    """

    def __init__(self, **overrides):
        self.seed = self._resolve("seed", "SEED", "42", overrides, _cast_int)
        self.backend = self._resolve("backend", "BACKEND", "sklearn", overrides, str)
        self.datasets = self._resolve(
            "datasets", "DATASETS", "blobs,moons,circles", overrides, _cast_str_list
        )
        self.data_seeds = self._resolve(
            "data_seeds", "DATA_SEEDS", "1,2,3", overrides, _cast_int_list
        )
        self.bags = self._resolve("bags", "BAGS", "8", overrides, _cast_int)
        self.max_size = self._resolve("max_size", "MAX_SIZE", "10", overrides, _cast_int)
        self.cv = self._resolve("cv", "CV", "3", overrides, _cast_int)
        self.n_samples = self._resolve("n_samples", "N_SAMPLES", "600", overrides, _cast_int)
        self.noise = self._resolve("noise", "NOISE", "0.10", overrides, _cast_float)
        self.patience = self._resolve("patience", "PATIENCE", "3", overrides, _cast_int)
        self.validate()

    @staticmethod
    def _resolve(attr, env_suffix, default, overrides, cast):
        if attr in overrides and overrides[attr] is not None:
            val = overrides[attr]
            # 列表型转换：override 已是 list 时直接透传（测试直接传列表）
            if cast in (_cast_str_list, _cast_int_list):
                return val if isinstance(val, list) else cast(val)
            # 仅对真实类型（int/float/str）做 isinstance 短路，自定义 cast 函数直接转换
            if cast in (int, float, str) and isinstance(val, cast):
                return val
            return cast(val)
        env_val = os.environ.get(_ENV_PREFIX + env_suffix)
        if env_val is not None:
            return cast(env_val)
        return cast(default)

    def validate(self):
        if self.seed < 0:
            raise ConfigError("seed 必须 >= 0")
        if self.backend not in ("sklearn", "numpy"):
            raise ConfigError(f"backend 必须 ∈ {{sklearn,numpy}}，收到 {self.backend!r}")
        if not self.datasets:
            raise ConfigError("datasets 不能为空")
        if not self.data_seeds:
            raise ConfigError("data_seeds 不能为空")
        for name, val in (
            ("bags", self.bags),
            ("max_size", self.max_size),
            ("cv", self.cv),
            ("n_samples", self.n_samples),
            ("patience", self.patience),
        ):
            if val <= 0:
                raise ConfigError(f"{name} 必须 > 0，收到 {val}")
        if not (0.0 <= self.noise < 1.0):
            raise ConfigError(f"noise 必须 ∈ [0,1)，收到 {self.noise}")
        return self

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "backend": self.backend,
            "datasets": self.datasets,
            "data_seeds": self.data_seeds,
            "bags": self.bags,
            "max_size": self.max_size,
            "cv": self.cv,
            "n_samples": self.n_samples,
            "noise": self.noise,
            "patience": self.patience,
        }
