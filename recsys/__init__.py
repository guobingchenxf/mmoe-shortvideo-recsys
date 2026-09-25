"""基于 MMoE 的实时短视频推荐引擎。

包结构：
    recsys.data       数据层：模拟日志生成、数据集 Schema
    recsys.features   特征工程层：FeatureColumn、离线特征流水线
    recsys.models     模型层：MMoE 多目标模型、损失、指标、训练
    recsys.services   在线服务层：FastAPI 推理接口、缓存、召回排序
    recsys.utils      通用工具：日志、计时、IO
"""

__version__ = "0.1.0"
