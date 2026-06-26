"""
Kimodo SOMA77 motion -> SMPLData converter.

The command-line entry point writes a SMPL/AMASS-style NPZ.  The pipeline
converter returns the same fields as an in-memory SMPLData dictionary.
"""

import argparse
import logging
import os
import tempfile

import numpy as np

logger = logging.getLogger(__name__)

try:
    from .base import BaseConverter, create_smpl_data
    from .kimodo.motion_io import load_kimodo_npz_as_torch
    from .kimodo.skeleton.registry import build_skeleton
    from .kimodo.smplx import AMASSConverter
except ImportError:  # Allows: python converters/kimodo_converter.py ...
    from base import BaseConverter, create_smpl_data
    from kimodo.motion_io import load_kimodo_npz_as_torch
    from kimodo.skeleton.registry import build_skeleton
    from kimodo.smplx import AMASSConverter


class KimodoConverter(BaseConverter):
    """Convert a 77-joint Kimodo NPZ to this repo's SMPLData dictionary."""

    def convert(self, input_path, **kwargs):
        """Convert a Kimodo SOMA77 NPZ into SMPLData.

        Args:
            input_path: Kimodo-generated ``.npz`` path.
            **kwargs:
                z_up: Convert Kimodo Y-up coordinates to SMPL/AMASS Z-up
                    coordinates. Defaults to ``True``.
                source_fps: Source frame rate. Defaults to 30 Hz, matching
                    Kimodo generation output.
                target_fps: Optional frame rate for simple integer-step
                    downsampling after conversion.

        Returns:
            dict: SMPLData with SMPL/AMASS NPZ fields.
        """
        z_up = kwargs.get("z_up", True)
        source_fps = kwargs.get("source_fps", None)
        target_fps = kwargs.get("target_fps", None)

        logger.info(
            "Starting Kimodo->SMPL conversion: input=%s, z_up=%s, source_fps=%s, target_fps=%s",
            input_path, z_up, source_fps, target_fps,
        )

        tmp_dir = tempfile.mkdtemp(prefix="kimodo_to_smpl_")
        smpl_npz_path = os.path.join(tmp_dir, "smpl_output.npz")

        try:
            convert_kimodo_to_smpl_npz(
                input_path,
                smpl_npz_path,
                source_fps=source_fps,
                z_up=z_up,
            )
            return load_smpl_npz_as_data(smpl_npz_path, target_fps=target_fps)
        finally:
            if os.path.exists(smpl_npz_path):
                os.remove(smpl_npz_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)


def convert_kimodo_to_smpl_npz(
    input_path: str,
    output_path: str,
    *,
    source_fps: float | None = None,
    z_up: bool = True,
) -> None:
    """Convert a 77-joint Kimodo NPZ to a SMPL/AMASS-style NPZ."""
    logger.info("Loading Kimodo NPZ: %s", input_path)
    data, num_joints = load_kimodo_npz_as_torch(input_path, ensure_complete=False)
    logger.info("Loaded %d joints, %d frames", num_joints, data["local_rot_mats"].shape[0])

    if num_joints != 77:
        raise ValueError(f"Kimodo->SMPL conversion only supports SOMA77 input; got J={num_joints}.")

    source_fps = 30.0 if source_fps is None else float(source_fps)
    logger.info("Using source FPS=%.1f, z_up=%s", source_fps, z_up)

    logger.info("Building SOMA77 skeleton")
    soma_skeleton = build_skeleton(77)

    logger.info("Converting SOMA77 -> SMPLX skeleton (retargeting joints)")
    smplx_local, smplx_skeleton = soma_skeleton.to_SMPLXSkeleton22(
        data["local_rot_mats"],
        data["root_positions"],
    )

    data["local_rot_mats"] = smplx_local
    logger.info("Running forward kinematics on SMPLX skeleton")
    data["global_rot_mats"], data["posed_joints"], _ = smplx_skeleton.fk(
        smplx_local,
        data["root_positions"],
    )

    logger.info("Converting to AMASS format and saving: %s", output_path)
    converter = AMASSConverter(fps=source_fps, skeleton=smplx_skeleton)
    converter.convert_save_npz(data, output_path, z_up=z_up)
    logger.info("Conversion complete: %s", output_path)


def load_smpl_npz_as_data(input_path: str, *, target_fps: float | None = None) -> dict:
    """Load a SMPL/AMASS-style NPZ into the canonical SMPLData dictionary."""
    logger.info("Loading SMPL NPZ: %s", input_path)
    with np.load(str(input_path), allow_pickle=True) as data:
        trans = np.asarray(data["trans"], dtype=np.float32)
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)
        pose_body = np.asarray(data["pose_body"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32)
        mocap_frame_rate = float(data["mocap_frame_rate"])
        metadata = {
            key: data[key]
            for key in data.files
            if key not in {"trans", "root_orient", "pose_body", "betas", "mocap_frame_rate"}
        }

    logger.info(
        "Loaded %d frames at %.1f FPS (duration=%.2fs)",
        len(trans), mocap_frame_rate, len(trans) / mocap_frame_rate,
    )

    if target_fps is not None and float(target_fps) < mocap_frame_rate:
        original_frame_count = len(trans)
        step = max(1, round(mocap_frame_rate / float(target_fps)))
        logger.info(
            "Downsampling from %.1f FPS to %.1f FPS (step=%d, %d -> %d frames)",
            mocap_frame_rate, target_fps, step, original_frame_count,
            original_frame_count // step,
        )
        trans = trans[::step]
        root_orient = root_orient[::step]
        pose_body = pose_body[::step]
        for key in ("pose_jaw", "pose_eye", "pose_hand"):
            if key in metadata and getattr(metadata[key], "shape", (0,))[0] == original_frame_count:
                metadata[key] = metadata[key][::step]
        mocap_frame_rate = float(target_fps)
        metadata["mocap_time_length"] = len(trans) / mocap_frame_rate

    return create_smpl_data(
        mocap_frame_rate=mocap_frame_rate,
        trans=trans,
        root_orient=root_orient,
        pose_body=pose_body,
        betas=betas,
        **metadata,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a 77-joint Kimodo NPZ to a SMPL/AMASS-style NPZ.",
    )
    parser.add_argument("--input", required=True, help="Input Kimodo SOMA77 NPZ path")
    parser.add_argument("--output", required=True, help="Output SMPL NPZ path")
    parser.add_argument(
        "--source-fps",
        "--fps",
        dest="source_fps",
        type=float,
        default=None,
        help="Source Kimodo motion frame rate in Hz (default: 30).",
    )
    parser.add_argument(
        "--no-z-up",
        action="store_true",
        help="Disable Kimodo Y-up to SMPL/AMASS Z-up coordinate conversion.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_argparser().parse_args()
    convert_kimodo_to_smpl_npz(
        args.input,
        args.output,
        source_fps=args.source_fps,
        z_up=not args.no_z_up,
    )
