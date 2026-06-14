"""
GVHMR video/results helpers.

The primary GVHMR artifact for this project is ``hmr4d_results.pt`` because GMR
can consume that file directly. Exporting GVHMR results to SMPL/AMASS-style NPZ
is kept as an explicit compatibility/debug path.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from .base import BaseConverter, create_smpl_data
except ImportError:  # Allows: python motions_to_smpl/gvhmr_converter.py ...
    from base import BaseConverter, create_smpl_data

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GVHMR_ROOT = PROJECT_ROOT / "motions_to_smpl" / "gvhmr"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "motion_data" / "gvhmr_hmr4d_pt"


class GVHMRConverter(BaseConverter):
    """Run GVHMR and optionally export results as this repo's SMPLData."""

    def run(self, input_path, **kwargs) -> Path:
        """Return a GVHMR ``hmr4d_results.pt`` path, running GVHMR when needed."""
        return run_gvhmr_to_results(
            input_path,
            output_root=kwargs.get("output_root"),
            static_cam=kwargs.get("static_cam", False),
            use_dpvo=kwargs.get("use_dpvo", False),
            f_mm=kwargs.get("f_mm"),
            verbose=kwargs.get("verbose", False),
            skip_gvhmr=kwargs.get("skip_gvhmr", False),
        )

    def convert(self, input_path, **kwargs):
        """Convert a video or ``hmr4d_results.pt`` file into SMPLData.

        This is not the preferred path for GVHMR -> GMR. Use ``run`` or
        ``run_gvhmr_to_results`` when the next step is robot retargeting.

        Args:
            input_path: Video path, or an existing ``hmr4d_results.pt``.
            **kwargs:
                output_root: GVHMR output root. Defaults to
                    ``motions_to_smpl/gvhmr/outputs/demo``.
                static_cam: Forward ``--static_cam`` to GVHMR.
                use_dpvo: Forward ``--use_dpvo`` to GVHMR.
                f_mm: Forward ``--f_mm`` to GVHMR.
                verbose: Forward ``--verbose`` to GVHMR.
                target_fps: Optional downsample target FPS.
                skip_gvhmr: Treat input as an existing ``hmr4d_results.pt``.

        Returns:
            dict: SMPLData with the standard ``trans``, ``root_orient``,
                ``pose_body``, ``betas`` and ``mocap_frame_rate`` fields.
        """
        target_fps = kwargs.get("target_fps")
        hmr4d_results = self.run(input_path, **kwargs)
        return load_gvhmr_results_as_data(hmr4d_results, target_fps=target_fps)


def run_gvhmr_to_results(
    input_path: str | os.PathLike,
    *,
    output_root: str | os.PathLike | None = None,
    static_cam: bool = False,
    use_dpvo: bool = False,
    f_mm: int | None = None,
    verbose: bool = False,
    skip_gvhmr: bool = False,
) -> Path:
    """Return a GVHMR ``hmr4d_results.pt`` path, running GVHMR when needed."""
    return resolve_hmr4d_results(
        input_path,
        output_root=output_root,
        static_cam=static_cam,
        use_dpvo=use_dpvo,
        f_mm=f_mm,
        verbose=verbose,
        skip_gvhmr=skip_gvhmr,
    )


def resolve_hmr4d_results(
    input_path: str | os.PathLike,
    *,
    output_root: str | os.PathLike | None = None,
    static_cam: bool = False,
    use_dpvo: bool = False,
    f_mm: int | None = None,
    verbose: bool = False,
    skip_gvhmr: bool = False,
) -> Path:
    """Return a GVHMR ``hmr4d_results.pt`` path, running GVHMR when needed."""
    input_path = Path(input_path)
    if skip_gvhmr or input_path.name == "hmr4d_results.pt" or input_path.suffix == ".pt":
        if not input_path.exists():
            raise FileNotFoundError(f"GVHMR result file not found: {input_path}")
        return input_path

    output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    output_root = output_root.expanduser().resolve()
    expected_result = output_root / input_path.stem / "hmr4d_results.pt"

    if expected_result.exists():
        logger.info("Using existing GVHMR result: %s", expected_result)
        return expected_result

    run_gvhmr(
        video_path=input_path,
        output_root=output_root,
        static_cam=static_cam,
        use_dpvo=use_dpvo,
        f_mm=f_mm,
        verbose=verbose,
    )

    if not expected_result.exists():
        raise FileNotFoundError(f"GVHMR did not create expected result: {expected_result}")
    return expected_result


def run_gvhmr(
    *,
    video_path: Path,
    output_root: Path,
    static_cam: bool = False,
    use_dpvo: bool = False,
    f_mm: int | None = None,
    verbose: bool = False,
) -> None:
    """Run the local GVHMR demo script for a video."""
    video_path = video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "run.py",
        f"--video={video_path}",
        f"--output_root={output_root}",
    ]
    if static_cam:
        cmd.append("--static_cam")
    if use_dpvo:
        cmd.append("--use_dpvo")
    if f_mm is not None:
        cmd.append(f"--f_mm={int(f_mm)}")
    if verbose:
        cmd.append("--verbose")

    logger.info("Running GVHMR: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=GVHMR_ROOT, check=True)


def load_gvhmr_results_as_data(
    results_path: str | os.PathLike,
    *,
    target_fps: float | None = None,
) -> dict:
    """Load GVHMR ``hmr4d_results.pt`` into the canonical SMPLData dictionary."""
    results_path = Path(results_path)
    logger.info("Loading GVHMR results: %s", results_path)
    pred = torch.load(results_path, map_location="cpu")
    smpl_params = pred["smpl_params_global"]

    trans = _to_numpy(smpl_params["transl"])
    root_orient = _to_numpy(smpl_params["global_orient"])
    pose_body = _to_numpy(smpl_params["body_pose"])
    betas = _to_numpy(smpl_params["betas"])

    if betas.ndim == 2:
        betas = betas[0]
    if betas.shape[0] < 16:
        betas = np.pad(betas, (0, 16 - betas.shape[0]))
    elif betas.shape[0] > 16:
        betas = betas[:16]

    fps = 30.0
    metadata = {
        "gender": np.array("neutral"),
        "surface_model_type": np.array("smplx"),
        "mocap_time_length": len(trans) / fps,
        "source": np.array("gvhmr"),
        "gvhmr_results_path": np.array(str(results_path)),
    }

    if target_fps is not None and float(target_fps) < fps:
        original_frame_count = len(trans)
        step = max(1, round(fps / float(target_fps)))
        logger.info(
            "Downsampling from %.1f FPS to %.1f FPS (step=%d, %d -> %d frames)",
            fps,
            target_fps,
            step,
            original_frame_count,
            len(trans[::step]),
        )
        trans = trans[::step]
        root_orient = root_orient[::step]
        pose_body = pose_body[::step]
        fps = float(target_fps)
        metadata["mocap_time_length"] = len(trans) / fps

    logger.info("Loaded %d frames at %.1f FPS", len(trans), fps)
    return create_smpl_data(
        mocap_frame_rate=fps,
        trans=trans,
        root_orient=root_orient,
        pose_body=pose_body,
        betas=betas,
        **metadata,
    )


def convert_gvhmr_to_smpl_npz(
    input_path: str,
    output_path: str,
    *,
    output_root: str | os.PathLike | None = None,
    static_cam: bool = False,
    use_dpvo: bool = False,
    f_mm: int | None = None,
    verbose: bool = False,
    target_fps: float | None = None,
    skip_gvhmr: bool = False,
) -> None:
    """Convert a video or GVHMR result file to a SMPL/AMASS-style NPZ."""
    hmr4d_results = run_gvhmr_to_results(
        input_path,
        output_root=output_root,
        static_cam=static_cam,
        use_dpvo=use_dpvo,
        f_mm=f_mm,
        verbose=verbose,
        skip_gvhmr=skip_gvhmr,
    )
    export_gvhmr_results_to_smpl_npz(hmr4d_results, output_path, target_fps=target_fps)


def export_gvhmr_results_to_smpl_npz(
    results_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    target_fps: float | None = None,
) -> None:
    """Export an existing GVHMR result file to a SMPL/AMASS-style NPZ."""
    smpl_data = load_gvhmr_results_as_data(results_path, target_fps=target_fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **smpl_data)
    logger.info("Saved SMPL NPZ: %s", output_path)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run GVHMR for a video and print the hmr4d_results.pt path. "
            "If --output is provided, also export a SMPL/AMASS-style NPZ."
        ),
    )
    parser.add_argument("--input", required=True, help="Input video path or GVHMR hmr4d_results.pt")
    parser.add_argument("--output", default=None, help="Optional output SMPL NPZ path")
    parser.add_argument(
        "--output-root",
        default=None,
        help="GVHMR output root (default: motion_data/gvhmr_hmr4d_pt).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Optional output frame rate. Values below 30 FPS downsample the GVHMR result.",
    )
    parser.add_argument(
        "-s",
        "--static-cam",
        action="store_true",
        help="Forward --static_cam to GVHMR.",
    )
    parser.add_argument(
        "--use-dpvo",
        action="store_true",
        help="Forward --use_dpvo to GVHMR.",
    )
    parser.add_argument(
        "--f-mm",
        type=int,
        default=None,
        help="Forward --f_mm to GVHMR.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward --verbose to GVHMR.",
    )
    parser.add_argument(
        "--skip-gvhmr",
        action="store_true",
        help="Treat --input as an existing hmr4d_results.pt file.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_argparser().parse_args()
    hmr4d_results = run_gvhmr_to_results(
        args.input,
        output_root=args.output_root,
        static_cam=args.static_cam,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
        verbose=args.verbose,
        skip_gvhmr=args.skip_gvhmr,
    )
    logger.info("GVHMR results: %s", hmr4d_results)
    print(hmr4d_results)
    if args.output:
        export_gvhmr_results_to_smpl_npz(
            hmr4d_results,
            args.output,
            target_fps=args.target_fps,
        )
