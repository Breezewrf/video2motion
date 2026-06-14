import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
BODY_MODELS_ROOT = REPO_ROOT / "assets" / "body_models"


def os_chdir_to_proj_root():
    """useful for running notebooks in different directories."""
    os.chdir(PROJ_ROOT)
