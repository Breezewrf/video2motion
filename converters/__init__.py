"""
converters: 多源运动数据 → SMPL 统一格式转换器

支持的数据源:
    - amass: AMASS 动捕数据集
    - kimodo: Kimodo 文本生成动作

注意:
    video 是输入类型，不是 SMPLData 转换器。当前视频路径默认使用
    GVHMR 生成 hmr4d_results.pt，并由 GMR 直接消费该文件。

用法:
    from converters import CONVERTER_REGISTRY, validate_smpl_data

    converter = CONVERTER_REGISTRY["amass"]()
    smpl_data = converter.convert("path/to/amass.npz")
"""

from .base import BaseConverter, validate_smpl_data, create_smpl_data
from .amass_converter import AMASSConverter
from .kimodo_converter import KimodoConverter


CONVERTER_REGISTRY = {
    "amass": AMASSConverter,
    "kimodo": KimodoConverter,
}

__all__ = [
    "BaseConverter",
    "AMASSConverter",
    "KimodoConverter",
    "CONVERTER_REGISTRY",
    "validate_smpl_data",
    "create_smpl_data",
]
