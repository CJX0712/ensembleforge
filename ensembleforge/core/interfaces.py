"""协议接口定义（运行期鸭子类型校验用）。作者：晨星。"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseModelProto(Protocol):
    """基础模型契约：所有库模型（sklearn 包装 / numpy 兜底）都实现此接口。"""

    name: str

    def fit(self, X, y): ...

    def predict(self, X): ...

    def predict_proba(self, X) -> list[list[float]]: ...


@runtime_checkable
class EnsembleProto(Protocol):
    """集成契约：fit_oof 在交叉验证 OOF 概率上做选型，predict 在测试概率上做组合。"""

    def fit_oof(self, oof_preds, y): ...

    def predict(self, test_preds_all): ...

    @property
    def members(self): ...
