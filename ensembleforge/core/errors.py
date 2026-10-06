"""错误码与异常定义（E100~E500）。作者：晨星。"""


class EnsembleForgeError(Exception):
    """所有 EnsembleForge 异常的基类。"""

    code = "E000"

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class ConfigError(EnsembleForgeError):
    """配置/环境错误。"""

    code = "E100"


class DataError(EnsembleForgeError):
    """数据生成/加载错误。"""

    code = "E200"


class ModelError(EnsembleForgeError):
    """模型构建/训练错误。"""

    code = "E300"


class EnsembleError(EnsembleForgeError):
    """集成选择/组合错误。"""

    code = "E400"


class ArtifactError(EnsembleForgeError):
    """IO / 产物读写错误。"""

    code = "E500"
