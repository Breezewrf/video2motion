"""
GVHMR (Video) → SMPLData 转换器

GVHMR 从单目视频中估计人体全局运动，输出 SMPL 参数的 npz 文件。
GitHub: zehongs/GVHMR

TODO: 用户需要自行实现以下步骤
"""

import numpy as np
from .base import BaseConverter, create_smpl_data


class GVHMRConverter(BaseConverter):
    """将视频通过 GVHMR 转为 SMPLData。"""

    def convert(self, input_path, **kwargs):
        """从视频提取人体运动并转为 SMPLData。

        Args:
            input_path: 视频文件路径 (.mp4, .avi 等)
            **kwargs:
                device: 推理设备，默认 "cuda:0"
                其他 GVHMR 特有参数

        Returns:
            dict: SMPLData

        实现步骤:
            1. 调用 GVHMR 对视频进行推理
            2. GVHMR 输出 npz 文件，包含:
               - trans: (N, 3) 全局平移
               - root_orient: (N, 3) 根节点旋转 (axis-angle)
               - pose_body: (N, 63) 身体关节旋转 (axis-angle)
               - betas: (16,) 体型参数
               - mocap_frame_rate: 帧率
            3. 返回标准化的 SMPLData
        """
        raise NotImplementedError(
            "GVHMR 转换器尚未实现。请按以下步骤完成:\n"
            "1. 安装 GVHMR: git clone https://github.com/zehongs/GVHMR\n"
            "2. 在此方法中调用 GVHMR 的推理接口\n"
            "3. 从 GVHMR 输出中提取 SMPL 参数\n"
            "4. 用 create_smpl_data() 构造返回值\n"
            "\n"
            "GVHMR 输出的 npz 通常包含:\n"
            "  trans (N,3), root_orient (N,3), pose_body (N,63), "
            "betas (16,), mocap_frame_rate"
        )
