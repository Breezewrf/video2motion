#!/usr/bin/env python3

"""Upload UniMotion/GVHMR assets to a Hugging Face repository.

This preserves the on-disk layout expected by the codebase:

- motions_to_smpl/gvhmr/inputs/checkpoints/<subfolders>
- motions_to_smpl/gvhmr/hmr4d/utils/body_model/*.pt, *.pts
- assets/body_models/<subfolders>

python -m pip install huggingface_hub
export HF_TOKEN=your_hf_token
python upload_to_hf.py \
	--repo-id your-org-or-user/your-repo \
	--private \
	--create-repo
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload GVHMR assets to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. org/name")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to HF_TOKEN env/login.")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of Hugging Face repository to upload to.",
    )
    parser.add_argument(
        "--checkpoints-source",
        default="motions_to_smpl/gvhmr/inputs/checkpoints",
        help="Local checkpoint root to upload recursively.",
    )
    parser.add_argument(
        "--checkpoints-target",
        default="motions_to_smpl/gvhmr/inputs/checkpoints",
        help="Target folder inside the HF repo for checkpoints.",
    )
    parser.add_argument(
        "--body-model-source",
        default="motions_to_smpl/gvhmr/hmr4d/utils/body_model",
        help="Local body-model folder to upload. Only .pt and .pts files are uploaded.",
    )
    parser.add_argument(
        "--body-model-target",
        default="motions_to_smpl/gvhmr/hmr4d/utils/body_model",
        help="Target folder inside the HF repo for body-model files.",
    )
    parser.add_argument(
        "--assets-source",
        default="assets/body_models",
        help="Local assets/body_models folder to upload recursively.",
    )
    parser.add_argument(
        "--assets-target",
        default="assets/body_models",
        help="Target folder inside the HF repo for assets/body_models.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the HF repo first if it does not exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the HF repo as private.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "huggingface_hub is required. Install it with `python -m pip install huggingface_hub`."
        ) from exc

    api = HfApi(token=args.token)

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )

    checkpoints_source = (project_root / args.checkpoints_source).resolve()
    if not checkpoints_source.exists():
        raise FileNotFoundError(f"Missing checkpoint folder: {checkpoints_source}")

    api.upload_folder(
        folder_path=str(checkpoints_source),
        path_in_repo=args.checkpoints_target.strip("/"),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    print(f"Uploaded {checkpoints_source} -> {args.checkpoints_target}")

    body_model_source = (project_root / args.body_model_source).resolve()
    if not body_model_source.exists():
        raise FileNotFoundError(f"Missing body model folder: {body_model_source}")

    api.upload_folder(
        folder_path=str(body_model_source),
        path_in_repo=args.body_model_target.strip("/"),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        allow_patterns=["*.pt", "*.pts"],
    )
    print(f"Uploaded {body_model_source} -> {args.body_model_target}")

    assets_source = (project_root / args.assets_source).resolve()
    if not assets_source.exists():
        raise FileNotFoundError(f"Missing assets folder: {assets_source}")

    api.upload_folder(
        folder_path=str(assets_source),
        path_in_repo=args.assets_target.strip("/"),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )
    print(f"Uploaded {assets_source} -> {args.assets_target}")


if __name__ == "__main__":
    main()