"""
motions_to_smpl: 多源运动数据 → SMPL 统一格式转换器

支持的数据源:
    - video (GVHMR): 从视频估计人体运动
    - amass: AMASS 动捕数据集
    - kimodo: Kimodo 文本生成动作

用法:
    from motions_to_smpl import CONVERTER_REGISTRY, validate_smpl_data

    converter = CONVERTER_REGISTRY["amass"]()
    smpl_data = converter.convert("path/to/amass.npz")
"""

from .base import BaseConverter, validate_smpl_data, create_smpl_data
from .gvhmr_converter import GVHMRConverter
from .amass_converter import AMASSConverter
from .kimodo_converter import KimodoConverter


CONVERTER_REGISTRY = {
    "video": GVHMRConverter,
    "amass": AMASSConverter,
    "kimodo": KimodoConverter,
}

__all__ = [
    "BaseConverter",
    "GVHMRConverter",
    "AMASSConverter",
    "KimodoConverter",
    "CONVERTER_REGISTRY",
    "validate_smpl_data",
    "create_smpl_data",
]
