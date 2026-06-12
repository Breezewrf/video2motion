"""
SMPLData 统一数据结构 + 转换器基类

所有数据源转换器都继承 BaseConverter，输出统一的 SMPLData 格式。
"""

import numpy as np
from abc import ABC, abstractmethod


# SMPLData 字典结构定义
# {
#     "trans": np.ndarray,              # (N, 3) 全局平移
#     "root_orient": np.ndarray,        # (N, 3) 根节点旋转 (axis-angle)
#     "pose_body": np.ndarray,          # (N, 63) 身体关节旋转 (21x3, axis-angle)
#     "betas": np.ndarray,              # (16,) 体型参数
#     "mocap_frame_rate": float,         # 帧率
# }

SMPL_KEYS = ["trans", "root_orient", "pose_body", "betas", "mocap_frame_rate"]

SMPL_SHAPES = {
    "trans": (-1, 3),
    "root_orient": (-1, 3),
    "pose_body": (-1, 63),
    "betas": (16,),
}


def validate_smpl_data(data):
    """校验 SMPLData 格式是否正确。返回 (is_valid, error_msg)。"""
    if not isinstance(data, dict):
        return False, "SMPLData 必须是 dict 类型"

    for key in SMPL_KEYS:
        if key not in data:
            return False, f"缺少必要字段: {key}"

    if not isinstance(data["mocap_frame_rate"], (int, float, np.integer, np.floating)):
        return False, f"mocap_frame_rate 必须是数字，当前类型: {type(data['mocap_frame_rate'])}"

    n_frames = data["trans"].shape[0]

    for key in ["trans", "root_orient", "pose_body"]:
        arr = data[key]
        if not isinstance(arr, np.ndarray):
            return False, f"{key} 必须是 np.ndarray"
        if arr.shape[0] != n_frames:
            return False, f"{key} 帧数不一致: {arr.shape[0]} vs {n_frames}"

    if data["trans"].shape != (n_frames, 3):
        return False, f"trans shape 应为 ({n_frames}, 3), 实际 {data['trans'].shape}"
    if data["root_orient"].shape != (n_frames, 3):
        return False, f"root_orient shape 应为 ({n_frames}, 3), 实际 {data['root_orient'].shape}"
    if data["pose_body"].shape != (n_frames, 63):
        return False, f"pose_body shape 应为 ({n_frames}, 63), 实际 {data['pose_body'].shape}"
    if data["betas"].shape != (16,):
        return False, f"betas shape 应为 (16,), 实际 {data['betas'].shape}"

    return True, "OK"


def create_smpl_data(mocap_frame_rate, trans, root_orient, pose_body, betas, **metadata):
    """创建并校验一个 SMPLData 字典。"""
    smpl_data = {
        "trans": np.asarray(trans, dtype=np.float32),
        "root_orient": np.asarray(root_orient, dtype=np.float32),
        "pose_body": np.asarray(pose_body, dtype=np.float32),
        "betas": np.asarray(betas, dtype=np.float32),
        "mocap_frame_rate": float(mocap_frame_rate),
    }
    smpl_data.update(metadata)
    valid, msg = validate_smpl_data(smpl_data)
    if not valid:
        raise ValueError(f"SMPLData 校验失败: {msg}")
    return smpl_data


class BaseConverter(ABC):
    """运动数据转换器基类。所有数据源转换器需继承此类。"""

    @abstractmethod
    def convert(self, input_path, **kwargs):
        """将输入数据转为 SMPLData 格式。

        Args:
            input_path: 输入文件或目录路径
            **kwargs: 各转换器特有的参数

        Returns:
            dict: SMPLData 格式的运动数据
        """
        raise NotImplementedError

    def validate(self, smpl_data):
        """校验转换结果。"""
        return validate_smpl_data(smpl_data)
