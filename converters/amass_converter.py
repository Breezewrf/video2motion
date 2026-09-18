"""
AMASS 数据集 → SMPLData 转换器

AMASS 存储的是 SMPL/SMPL-X 风格的动捕数据（.npz 文件）。
主要字段: trans, root_orient, pose_body, betas, mocap_frame_rate
"""

import numpy as np
from .base import BaseConverter, create_smpl_data


class AMASSConverter(BaseConverter):
    """将 AMASS .npz 数据转为统一 SMPLData 格式。"""

    def convert(self, input_path, **kwargs):
        """加载 AMASS npz 文件并转为 SMPLData。

        Args:
            input_path: AMASS .npz 文件路径
            **kwargs:
                target_fps: 目标帧率，若不等于原始帧率则做简单降采样
                target_fps: 目标帧率，若不等于原始帧率则做简单降采样

        Returns:
            dict: SMPLData
        """
        target_fps = kwargs.get("target_fps", None)

        with np.load(str(input_path), allow_pickle=True) as data:
            # 提取 SMPL/AMASS 字段
            trans = data["trans"]              # (N, 3)
            root_orient = data["root_orient"]  # (N, 3)
            pose_body = data["pose_body"]      # (N, 63) 21 joints x 3
            betas = data["betas"]              # (16,)
            fps = float(data["mocap_frame_rate"])
            metadata = {
                key: data[key]
                for key in data.files
                if key not in {"trans", "root_orient", "pose_body", "betas", "mocap_frame_rate"}
            }

        # 简单降采样（如果需要）
        if target_fps is not None and target_fps < fps:
            original_frame_count = len(trans)
            step = max(1, round(fps / target_fps))
            trans = trans[::step]
            root_orient = root_orient[::step]
            pose_body = pose_body[::step]
            for key in ("pose_jaw", "pose_eye", "pose_hand"):
                if key in metadata and getattr(metadata[key], "shape", (0,))[0] == original_frame_count:
                    metadata[key] = metadata[key][::step]
            fps = float(target_fps)
            metadata["mocap_time_length"] = len(trans) / fps

        smpl_data = create_smpl_data(
            mocap_frame_rate=fps,
            trans=trans,
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas,
            **metadata,
        )

        return smpl_data
