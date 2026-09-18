"""
UniMotion processing pipeline.

Main routes:
    video --video-method gvhmr -> hmr4d_results.pt -> GMR robot pkl
    kimodo/amass -> SMPL/SMPL-X npz -> GMR robot pkl
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

from converters import CONVERTER_REGISTRY
from converters.gvhmr_converter import run_gvhmr_to_results, export_gvhmr_results_to_smpl_npz

PROJECT_ROOT = Path(__file__).resolve().parent
MOTION_DATA_ROOT = PROJECT_ROOT / "motion_data"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_to_smpl_npz(source: str, input_path: Path, output_path: Path, *, target_fps=None) -> Path:
    """Convert a SMPL-compatible source to a SMPL/AMASS-style NPZ."""
    if source not in CONVERTER_REGISTRY:
        raise ValueError(f"Unsupported source: {source}. Available: {list(CONVERTER_REGISTRY)}")

    converter = CONVERTER_REGISTRY[source]()
    smpl_data = converter.convert(str(input_path), target_fps=target_fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **smpl_data)
    logger.info("Saved SMPL NPZ: %s", output_path)
    return output_path


def retarget_smplx_npz_to_robot(
    smplx_file: Path,
    robot: str,
    save_path: Path,
    *,
    record_video: bool = False,
    rate_limit: bool = False,
) -> Path:
    """Run GMR on a SMPL/SMPL-X NPZ file."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "gmr" / "scripts" / "smplx_to_robot.py"),
        "--smplx_file",
        str(smplx_file),
        "--robot",
        robot,
        "--save_path",
        str(save_path),
    ]
    if record_video:
        cmd.append("--record_video")
    if rate_limit:
        cmd.append("--rate_limit")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running GMR SMPLX retarget: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return save_path


def retarget_gvhmr_results_to_robot(
    hmr4d_results: Path,
    robot: str,
    save_path: Path,
    *,
    record_video: bool = False,
    rate_limit: bool = False,
) -> Path:
    """Run GMR directly on GVHMR hmr4d_results.pt."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "gmr" / "scripts" / "gvhmr_to_robot.py"),
        "--gvhmr_pred_file",
        str(hmr4d_results),
        "--robot",
        robot,
        "--save_path",
        str(save_path),
    ]
    if record_video:
        cmd.append("--record_video")
    if rate_limit:
        cmd.append("--rate_limit")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running GMR GVHMR retarget: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return save_path


def convert_gmr_pkl_to_csv(pkl_path: Path, output_dir: Path | None = None) -> Path:
    """Convert one GMR pickle file to CSV for mjlab."""
    output_dir = output_dir or pkl_path.parent / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pkl_path.stem}.csv"

    with open(pkl_path, "rb") as f:
        motion_data = joblib.load(f)

    dof_pos = motion_data["dof_pos"]
    motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
    motion[:, :3] = motion_data["root_pos"]
    motion[:, 3:7] = motion_data["root_rot"]
    motion[:, 7:] = dof_pos

    frame_rate = float(motion_data["fps"])
    if frame_rate > 30:
        downsample_factor = frame_rate / 30.0
        indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
        motion = motion[indices]
        logger.info("Downsampled CSV motion from %.1f FPS to 30 FPS", frame_rate)

    np.savetxt(output_path, motion, delimiter=",")
    logger.info("Saved CSV: %s", output_path)
    return output_path


def convert_csv_to_mjlab_npz(
    csv_path: Path,
    *,
    output_name: str,
    mjlab_robot: str,
    input_fps: float,
    output_fps: float,
    device: str,
    render: bool,
) -> Path:
    """Convert CSV motion to unitree_rl_mjlab motion NPZ."""
    cmd = [
        sys.executable,
        "scripts/csv_to_npz.py",
        "--robot",
        mjlab_robot,
        "--input-file",
        str(csv_path),
        "--output-name",
        output_name,
        "--input-fps",
        str(input_fps),
        "--output-fps",
        str(output_fps),
        "--device",
        device,
    ]
    if render:
        cmd.append("--render")

    mjlab_root = PROJECT_ROOT / "unitree_rl_mjlab"
    logger.info("Running mjlab CSV->NPZ: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=mjlab_root, check=True)
    if not output_name.endswith(".npz"):
        output_name = f"{output_name}.npz"
    return mjlab_root / "src" / "assets" / "motions" / mjlab_robot / output_name


def visualize_robot_motion(pkl_path: Path, robot: str, *, record_video=False, video_path=None) -> None:
    """Launch the UniMotion robot motion viewer."""
    from data_loader import load_robot_motion
    from viewer import RobotMotionViewer

    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(pkl_path, xyzw=True)
    env = RobotMotionViewer(
        robot_type=robot,
        motion_fps=fps,
        record_video=record_video,
        video_path=video_path,
    )
    try:
        frame_idx = 0
        while True:
            if frame_idx >= root_pos.shape[0]:
                frame_idx = 0
            env.step(root_pos[frame_idx], root_rot[frame_idx], dof_pos[frame_idx])
            frame_idx += 1
    except KeyboardInterrupt:
        logger.info("Viewer interrupted")
    finally:
        env.close()


def infer_name(input_path: Path, explicit_name: str | None) -> str:
    return explicit_name or input_path.stem


def default_robot_motion_root(robot: str) -> Path:
    return MOTION_DATA_ROOT / robot


def default_gmr_pkl_dir(robot: str) -> Path:
    return default_robot_motion_root(robot) / "gmr_pkl"


def default_smpl_npz_dir(robot: str) -> Path:
    return default_robot_motion_root(robot) / "smpl_npz"


def default_gvhmr_output_root(robot: str) -> Path:
    return default_robot_motion_root(robot) / "gvhmr_hmr4d_pt"


def validate_video_input(input_path: Path) -> None:
    if input_path.name == "hmr4d_results.pt" or input_path.suffix == ".pt":
        raise ValueError(
            "pipeline.py expects a real video file for --source video. "
            "Use gmr/scripts/gvhmr_to_robot.py directly for an existing hmr4d_results.pt."
        )
    if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {input_path.suffix}. Expected one of {sorted(VIDEO_EXTENSIONS)}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniMotion source -> GMR robot motion pipeline")
    parser.add_argument("--source", required=True, choices=["video", "kimodo", "amass"])
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--name", default=None, help="Output basename (default: input stem)")
    parser.add_argument("--robot", default="unitree_g1_23dof", help="GMR target robot")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for GMR pkl outputs (default: motion_data/<robot>/gmr_pkl)",
    )
    parser.add_argument(
        "--smpl-output-dir",
        default=None,
        help="Directory for SMPL npz outputs (default: motion_data/<robot>/smpl_npz)",
    )

    parser.add_argument("--video-method", default="gvhmr", choices=["gvhmr"], help="Video-to-motion method")
    parser.add_argument(
        "--gvhmr-output-root",
        default=None,
        help="GVHMR output root (default: motion_data/<robot>/gvhmr_hmr4d_pt)",
    )
    parser.add_argument("-s", "--static-cam", action="store_true", help="Forward --static_cam to GVHMR")
    parser.add_argument("--use-dpvo", action="store_true", help="Forward --use_dpvo to GVHMR")
    parser.add_argument("--f-mm", type=int, default=None, help="Forward --f_mm to GVHMR")
    parser.add_argument("--verbose-gvhmr", action="store_true", help="Forward --verbose to GVHMR")
    parser.add_argument("--export-gvhmr-smpl", action="store_true", help="Also export GVHMR result to SMPL npz")

    parser.add_argument("--target-fps", type=float, default=None, help="Optional SMPL export target FPS")
    parser.add_argument("--record-gmr-video", action="store_true", help="Record GMR retarget viewer video")
    parser.add_argument("--rate-limit", action="store_true", help="Rate-limit the GMR viewer loop")
    parser.add_argument("--skip-gmr", action="store_true", help="Skip retargeting and use --gmr-output")
    parser.add_argument("--gmr-output", default=None, help="Existing GMR pkl path or explicit output pkl path")

    parser.add_argument("--to-csv", action="store_true", help="Convert GMR pkl to CSV")
    parser.add_argument("--to-mjlab-npz", action="store_true", help="Convert CSV to unitree_rl_mjlab NPZ")
    parser.add_argument("--mjlab-robot", default="g1_23dof", choices=["g1", "g1_23dof"])
    parser.add_argument("--mjlab-input-fps", type=float, default=30.0)
    parser.add_argument("--mjlab-output-fps", type=float, default=30.0)
    parser.add_argument("--mjlab-device", default="cuda:0")
    parser.add_argument("--mjlab-render", action="store_true")

    parser.add_argument("--visualize", action="store_true", help="Visualize final GMR pkl in UniMotion")
    parser.add_argument("--viewer-record-video", action="store_true")
    parser.add_argument("--viewer-video-path", default="videos/example.mp4")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_path = Path(args.input)
    name = infer_name(input_path, args.name)
    robot_motion_root = default_robot_motion_root(args.robot)
    output_dir = Path(args.output_dir) if args.output_dir else robot_motion_root / "gmr_pkl"
    smpl_output_dir = Path(args.smpl_output_dir) if args.smpl_output_dir else robot_motion_root / "smpl_npz"
    gvhmr_output_root = Path(args.gvhmr_output_root) if args.gvhmr_output_root else robot_motion_root / "gvhmr_hmr4d_pt"
    gmr_pkl = Path(args.gmr_output) if args.gmr_output else output_dir / f"{name}.pkl"

    if args.source == "video":
        validate_video_input(input_path)

    if args.skip_gmr:
        if not gmr_pkl.exists():
            raise FileNotFoundError(f"--skip-gmr requires an existing --gmr-output path: {gmr_pkl}")
    elif args.source == "video":
        if args.video_method == "gvhmr":
            hmr4d_results = run_gvhmr_to_results(
                input_path,
                output_root=gvhmr_output_root,
                static_cam=args.static_cam,
                use_dpvo=args.use_dpvo,
                f_mm=args.f_mm,
                verbose=args.verbose_gvhmr,
            )
            logger.info("GVHMR result ready: %s", hmr4d_results)
            if args.export_gvhmr_smpl:
                smpl_npz = smpl_output_dir / f"{name}.npz"
                export_gvhmr_results_to_smpl_npz(hmr4d_results, smpl_npz, target_fps=args.target_fps)
            retarget_gvhmr_results_to_robot(
                hmr4d_results,
                args.robot,
                gmr_pkl,
                record_video=args.record_gmr_video,
                rate_limit=args.rate_limit,
            )
        else:
            raise ValueError(f"Unsupported video method: {args.video_method}")
    else:
        smpl_npz = smpl_output_dir / f"{name}.npz"
        if not smpl_npz.exists():
            convert_to_smpl_npz(args.source, input_path, smpl_npz, target_fps=args.target_fps)
        else:
            logger.info("Using existing SMPL NPZ: %s", smpl_npz)
        retarget_smplx_npz_to_robot(
            smpl_npz,
            args.robot,
            gmr_pkl,
            record_video=args.record_gmr_video,
            rate_limit=args.rate_limit,
        )

    logger.info("GMR pkl ready: %s", gmr_pkl)

    csv_path = None
    if args.to_csv or args.to_mjlab_npz:
        csv_path = convert_gmr_pkl_to_csv(gmr_pkl)

    if args.to_mjlab_npz:
        mjlab_npz = convert_csv_to_mjlab_npz(
            csv_path,
            output_name=f"{name}.npz",
            mjlab_robot=args.mjlab_robot,
            input_fps=args.mjlab_input_fps,
            output_fps=args.mjlab_output_fps,
            device=args.mjlab_device,
            render=args.mjlab_render,
        )
        logger.info("mjlab NPZ ready: %s", mjlab_npz)

    if args.visualize:
        visualize_robot_motion(
            gmr_pkl,
            args.robot,
            record_video=args.viewer_record_video,
            video_path=args.viewer_video_path,
        )


if __name__ == "__main__":
    main()
