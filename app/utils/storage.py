import os
from pathlib import Path
import shutil

TMP_DIR = "tmp_videos"
os.makedirs(TMP_DIR, exist_ok=True)


def fetch_video(path: str) -> str:
    """
    For now:
    - if local → return path
    - later → download from S3
    """

    if path.startswith("s3://"):
        # TODO
        raise NotImplementedError("S3 download not added yet")

    return path