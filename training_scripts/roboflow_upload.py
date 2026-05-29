"""
roboflow_upload.py — upload the filtered COCO dataset to a Roboflow project

Reads image paths from data/train.txt and data/val.txt (produced by `python train.py prepare`)
and uploads each image with its YOLO label to Roboflow using parallel threads.

Successfully uploaded images are logged to data/uploaded.txt so that re-running
the script after a failure will skip already-uploaded images automatically.

Credentials are loaded from .env in the project root, which should contain:
    ROBOFLOW_API_KEY=...
    ROBOFLOW_WORKSPACE=...
    ROBOFLOW_PROJECT=...

Usage:
    python roboflow_upload.py                     # upload train + val (16 threads)
    python roboflow_upload.py --split train        # train only
    python roboflow_upload.py --split valid        # val only
    python roboflow_upload.py --batch "coco_v1"   # group into a named batch
    python roboflow_upload.py --workers 32         # more threads = faster (watch rate limits)
"""

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

API_KEY      = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE_ID = os.environ.get("ROBOFLOW_WORKSPACE")
PROJECT_ID   = os.environ.get("ROBOFLOW_PROJECT")

UPLOAD_LOG = ROOT / "data" / "uploaded.txt"


def label_path_for(img_path: Path) -> Path:
    return Path(str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")


def load_uploaded() -> set:
    if UPLOAD_LOG.exists():
        return set(UPLOAD_LOG.read_text().splitlines())
    return set()


def _upload_one(args):
    project, img_path, split, batch_name, retries, log_lock, log_file = args
    lbl = label_path_for(img_path)
    kwargs = dict(
        image_path        = str(img_path),
        split             = split,
        num_retry_uploads = retries,
    )
    if batch_name:
        kwargs["batch_name"] = batch_name
    if lbl.exists() and lbl.stat().st_size > 0:
        kwargs["annotation_path"] = str(lbl)

    project.upload(**kwargs)

    with log_lock:
        with open(log_file, "a") as f:
            f.write(str(img_path) + "\n")


def upload_split(project, txt_file: Path, split: str, batch_name, retries: int, workers: int, already_uploaded: set):
    if not txt_file.exists():
        print(f"  {txt_file.name} not found — skipping.")
        return 0, 0, 0

    all_paths = [Path(p) for p in txt_file.read_text().splitlines() if p]
    paths     = [p for p in all_paths if str(p) not in already_uploaded]
    skipped   = len(all_paths) - len(paths)

    if skipped:
        print(f"  {split}: {len(paths):,} remaining ({skipped:,} already uploaded, skipping)")
    else:
        print(f"  {split}: {len(paths):,} images")

    if not paths:
        return 0, 0, skipped

    log_lock = threading.Lock()
    uploaded = failed = 0
    tasks = [(project, img, split, batch_name, retries, log_lock, UPLOAD_LOG) for img in paths]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_upload_one, t): t[1] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=split, unit="img"):
            try:
                future.result()
                uploaded += 1
            except Exception as e:
                print(f"\n  Failed: {futures[future].name} — {e}")
                failed += 1

    return uploaded, failed, skipped


def main(split="both", batch_name=None, retries=3, workers=16):
    if not API_KEY:
        print("ERROR: ROBOFLOW_API_KEY not found. Create a .env file in the project root.")
        print("  ROBOFLOW_API_KEY=your_key")
        print("  ROBOFLOW_WORKSPACE=your_workspace")
        print("  ROBOFLOW_PROJECT=your_project")
        return

    from roboflow import Roboflow

    already_uploaded = load_uploaded()
    if already_uploaded:
        print(f"Resuming — {len(already_uploaded):,} images already logged in {UPLOAD_LOG.name}")

    print(f"Connecting to {WORKSPACE_ID}/{PROJECT_ID} ...")
    project = Roboflow(api_key=API_KEY).workspace(WORKSPACE_ID).project(PROJECT_ID)

    total_uploaded = total_failed = total_skipped = 0

    if split in ("both", "train"):
        u, f, s = upload_split(project, ROOT / "data" / "train.txt", "train", batch_name, retries, workers, already_uploaded)
        total_uploaded += u; total_failed += f; total_skipped += s

    if split in ("both", "valid"):
        u, f, s = upload_split(project, ROOT / "data" / "val.txt", "valid", batch_name, retries, workers, already_uploaded)
        total_uploaded += u; total_failed += f; total_skipped += s

    print(f"\nDone: {total_uploaded:,} uploaded, {total_skipped:,} skipped, {total_failed} failed")
    if total_failed:
        print(f"  Re-run the script to retry the {total_failed} failed images.")
    print(f"View at: https://app.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",   default="both",  choices=["both", "train", "valid"])
    p.add_argument("--batch",   default=None,    help="Roboflow batch name")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=16, help="parallel upload threads")
    args = p.parse_args()
    main(args.split, args.batch, args.retries, args.workers)
