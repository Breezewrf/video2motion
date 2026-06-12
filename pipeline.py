"""
多源运动数据统一管线: 输入 → SMPL → GMR → UniMotion 可视化

用法:
    python pipeline.py --source video --input /path/to/video.mp4 --robot unitree_g1
    python pipeline.py --source amass --input /path/to/amass.npz --robot unitree_g1
    python pipeline.py --source kimodo --input /path/to/kimodo_output --robot unitree_g1
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

from motions_to_smpl import CONVERTER_REGISTRY, validate_smpl_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Step 1: 各数据源 → SMPL
# ============================================================

def convert_to_smpl(source, input_path, **kwargs):
    """根据数据源类型，将输入转为 SMPLData。"""
    if source not in CONVERTER_REGISTRY:
        raise ValueError(f"不支持的数据源: {source}，可选: {list(CONVERTER_REGISTRY.keys())}")

    converter = CONVERTER_REGISTRY[source]()
    logger.info(f"使用 {converter.__class__.__name__} 转换: {input_path}")
    smpl_data = converter.convert(input_path, **kwargs)

    valid, msg = validate_smpl_data(smpl_data)
    if not valid:
        raise ValueError(f"SMPLData 校验失败: {msg}")

    logger.info(
        f"SMPL 转换完成: {smpl_data['trans'].shape[0]} 帧, "
        f"{smpl_data['mocap_frame_rate']} fps"
    )
    return smpl_data


# ============================================================
# Step 2: SMPL → GMR 重定向 (输出机器人运动 pickle)
# ============================================================

def run_gmr_retarget(smpl_data, robot, output_dir, gmr_path=None):
    """调用 GMR 将 SMPL 运动重定向到目标机器人。

    TODO: 用户需要根据自己的 GMR 安装方式实现此函数。

    Args:
        smpl_data: SMPLData 字典
        robot: 目标机器人名称 (如 "unitree_g1")
        output_dir: 输出目录
        gmr_path: GMR 仓库路径 (可选)

    Returns:
        str: GMR 输出的 pickle 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_pkl = os.path.join(output_dir, f"gmr_{robot}_{int(time.time())}.pkl")

    # ---------------------------------------------------------
    # TODO: 在此处调用 GMR 重定向
    #
    # 方式 A: 通过 subprocess 调用 GMR 命令行
    #   import subprocess
    #   cmd = [
    #       "python", os.path.join(gmr_path, "retarget.py"),
    #       "--smpl", smpl_npz_path,
    #       "--robot", robot,
    #       "--output", output_pkl,
    #   ]
    #   subprocess.run(cmd, check=True)
    #
    # 方式 B: 直接 import GMR 模块
    #   sys.path.insert(0, gmr_path)
    #   from gmr.retarget import retarget
    #   retarget(smpl_data, robot, output_pkl)
    #
    # 方式 C: GMR 已经输出了 pickle，直接复制/引用
    #   output_pkl = existing_gmr_output_path
    # ---------------------------------------------------------

    raise NotImplementedError(
        "GMR 重定向尚未实现。请按以下步骤完成:\n"
        "1. 安装 GMR: git clone https://github.com/zhengyiluo/GMR\n"
        "2. 在此函数中调用 GMR 的重定向接口\n"
        "3. GMR 会输出一个 pickle 文件，包含:\n"
        "   fps, root_pos (N,3), root_rot (N,4), dof_pos (N, num_dofs)\n"
        "4. 返回该 pickle 文件路径\n"
        f"\n期望输出路径: {output_pkl}"
    )


# ============================================================
# Step 3: GMR 输出 → UniMotion 可视化
# ============================================================

def visualize_robot_motion(pkl_path, robot, record_video=False, video_path=None):
    """加载 GMR 输出的 pickle 并启动 UniMotion 可视化。"""
    from data_loader import load_robot_motion
    from viewer import RobotMotionViewer

    motion_data, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(pkl_path, xyzw=True)

    env = RobotMotionViewer(
        robot_type=robot,
        motion_fps=fps,
        record_video=record_video,
        video_path=video_path,
    )

    frame_idx = 0
    num_frames = root_pos.shape[0]
    logger.info(f"开始可视化: {num_frames} 帧, {fps} fps, 机器人: {robot}")

    try:
        while True:
            if frame_idx >= num_frames:
                frame_idx = 0
            env.step(root_pos[frame_idx], root_rot[frame_idx], dof_pos[frame_idx])
            frame_idx += 1
    except KeyboardInterrupt:
        logger.info("可视化被用户中断")
    finally:
        env.close()


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="多源运动数据统一管线: 输入 → SMPL → GMR → UniMotion 可视化"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        choices=list(CONVERTER_REGISTRY.keys()),
        help="数据源类型: video (GVHMR), amass, kimodo"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="输入文件或目录路径"
    )
    parser.add_argument(
        "--robot", type=str, default="unitree_g1",
        help="目标机器人 (默认: unitree_g1)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="中间产物输出目录 (默认: outputs/)"
    )
    parser.add_argument(
        "--gmr_path", type=str, default=None,
        help="GMR 仓库路径 (可选)"
    )
    parser.add_argument(
        "--skip_gmr", action="store_true",
        help="跳过 GMR，直接使用已有的 GMR 输出 pickle 文件 (调试用)"
    )
    parser.add_argument(
        "--gmr_output", type=str, default=None,
        help="配合 --skip_gmr 使用，指定已有的 GMR 输出 pickle 路径"
    )
    parser.add_argument(
        "--save_smpl", action="store_true",
        help="将中间 SMPL 数据保存为 npz 文件"
    )
    parser.add_argument(
        "--record_video", action="store_true",
        help="录制可视化视频"
    )
    parser.add_argument(
        "--video_path", type=str, default="videos/example.mp4",
        help="录制视频输出路径 (默认: videos/example.mp4)"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------
    # Step 1: 转换为 SMPL
    # -----------------------------------------------------------
    if args.skip_gmr:
        logger.info("跳过 SMPL 转换 (--skip_gmr)")
        gmr_pkl = args.gmr_output
        if not gmr_pkl or not os.path.exists(gmr_pkl):
            logger.error("--skip_gmr 模式需要通过 --gmr_output 指定有效的 pickle 文件路径")
            sys.exit(1)
    else:
        smpl_data = convert_to_smpl(args.source, args.input)

        # 可选: 保存中间 SMPL 数据
        if args.save_smpl:
            smpl_path = os.path.join(args.output_dir, "smpl_data.npz")
            os.makedirs(args.output_dir, exist_ok=True)
            np.savez(smpl_path, **smpl_data)
            logger.info(f"SMPL 数据已保存: {smpl_path}")

        # -----------------------------------------------------------
        # Step 2: GMR 重定向
        # -----------------------------------------------------------
        gmr_pkl = run_gmr_retarget(
            smpl_data, args.robot, args.output_dir, args.gmr_path
        )

    # -----------------------------------------------------------
    # Step 3: 可视化
    # -----------------------------------------------------------
    visualize_robot_motion(
        gmr_pkl, args.robot,
        record_video=args.record_video,
        video_path=args.video_path,
    )


if __name__ == "__main__":
    main()
