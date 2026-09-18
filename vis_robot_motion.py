from viewer import RobotMotionViewer
from data_loader import load_robot_motion
import argparse
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reorder_dof_pos(dof_pos, source_order, target_order):
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D, got shape {dof_pos.shape}")
    if dof_pos.shape[1] != len(source_order):
        raise ValueError(
            f"Motion dof_pos has {dof_pos.shape[1]} columns, but source order has {len(source_order)} names"
        )

    source_index = {name: idx for idx, name in enumerate(source_order)}
    try:
        target_indices = [source_index[name] for name in target_order]
    except KeyError as exc:
        raise ValueError(f"Joint name {exc.args[0]!r} not found in motion source order") from exc
    return dof_pos[:, target_indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
    parser.add_argument("--joint_sequence_type", type=str, default="mujoco", choices=["mujoco", "isaac"])
    parser.add_argument("--xyzw", action="store_true", help="Whether the root rotation in the motion data is in xyzw format (True) or wxyz format (False, default). If True, will convert to wxyz format for Mujoco viewer." \
    "Note: wxyz is for Mujoco, IsaacSim and IsaacLab")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, _, _ = load_robot_motion(robot_motion_path, xyzw=args.xyzw)

    if args.joint_sequence_type == "isaac":
        logger.info("Converting joint sequence from Isaac Lab format to Mujoco format for the viewer")
        import yaml
        with open("mapping.yaml", "r") as f:
            mapping_data = yaml.safe_load(f)
        isaaclab_dof_names = mapping_data["isaaclab_dof_names"]
        mujoco_dof_names = mapping_data["mujoco_dof_names"]

        motion_dof_pos = reorder_dof_pos(
            motion_dof_pos,
            source_order=isaaclab_dof_names,
            target_order=mujoco_dof_names,
        )

    if motion_root_pos.shape[0] != motion_root_rot.shape[0] or motion_root_pos.shape[0] != motion_dof_pos.shape[0]:
        raise ValueError(
            "root_pos, root_rot, and dof_pos must have the same number of frames: "
            f"{motion_root_pos.shape[0]}, {motion_root_rot.shape[0]}, {motion_dof_pos.shape[0]}"
        )
      
    logger.info(f"Loaded motion data from {robot_motion_path}, motion_root_pos: {motion_root_pos.shape}, motion_root_rot: {motion_root_rot.shape}, motion_dof_pos: {motion_dof_pos.shape}, motion_fps: {motion_fps}")
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    while True:
        env.step(motion_root_pos[frame_idx], 
                motion_root_rot[frame_idx], 
                motion_dof_pos[frame_idx], 
                rate_limit=True)
        frame_idx += 1
        if frame_idx >= len(motion_root_pos):
            frame_idx = 0
    env.close()