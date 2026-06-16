#!/usr/bin/env python3

"""Download the shared UniMotion/GVHMR assets from Hugging Face.

This restores the same local paths used by the uploader:

- motions_to_smpl/gvhmr/inputs/checkpoints/<subfolders>
- motions_to_smpl/gvhmr/hmr4d/utils/body_model/*.pt, *.pts
- assets/body_models/<subfolders>

Usage: python3 download_from_hf.py   --repo-id breezewrf/unimotion_ckpt --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
from pathlib import Path


ALLOW_PATTERNS = [
    "motions_to_smpl/gvhmr/inputs/checkpoints/**",
    "motions_to_smpl/gvhmr/hmr4d/utils/body_model/*.pt",
    "motions_to_smpl/gvhmr/hmr4d/utils/body_model/*.pts",
    "assets/body_models/**",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UniMotion/GVHMR assets from Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. org/name")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env/login.")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of Hugging Face repository to download from.",
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Directory to populate. Defaults to the repo root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.target_dir).resolve() if args.target_dir else Path(__file__).resolve().parent

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "huggingface_hub is required. Install it with `python -m pip install huggingface_hub`."
        ) from exc

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=args.token,
        local_dir=str(project_root),
        allow_patterns=ALLOW_PATTERNS,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded assets into {project_root}")


if __name__ == "__main__":
    main()