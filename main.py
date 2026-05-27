"""
main.py — YOLO26 football detector

Usage:
    python main.py prepare
    python main.py train --size n --epochs 50
    python main.py validate --size n
    python main.py export --size n
    python main.py track --video data/clip.mp4
    python main.py csv --video data/clip.mp4
    python main.py label --images data/my_images
    python main.py stats --csv runs/detections.csv
    python main.py collect
"""

import argparse
import csv
import glob
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

# ── Repo root (all paths relative to this file) ────────────────────────────────
ROOT = Path(__file__).parent

# ── Dataset constants ──────────────────────────────────────────────────────────
COCO_ROOT   = ROOT / "Datasets" / "coco2017"
COCO_KEEP   = {1: 0, 37: 1}                   # person, sports_ball
CLASS_NAMES = {0: "person", 1: "sports_ball"}


def get_device():
    if torch.cuda.is_available():
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
        return 0
    if torch.backends.mps.is_available():
        print("[device] MPS — Apple Silicon")
        return "mps"
    print("[device] CPU")
    return "cpu"


def best_weights(size):
    return ROOT / "runs" / f"yolo26{size}_football" / "weights" / "best.pt"


# ── Functions ──────────────────────────────────────────────────────────────────

def prepare():
    """Filter COCO annotations → YOLO labels + football.yaml."""
    labels_root = ROOT / "data" / "labels"
    yaml_path   = ROOT / "data" / "football.yaml"

    def build(ann_file, img_dir, label_dir, list_out):
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f"[prepare] Reading {ann_file.name} ...", flush=True)
        data = json.loads(ann_file.read_text())
        print(f"[prepare] Indexing {len(data['annotations']):,} annotations ...", flush=True)
        img_info = {i["id"]: i for i in data["images"]}
        by_image = defaultdict(list)
        for ann in data["annotations"]:
            if ann["category_id"] in COCO_KEEP and not ann.get("iscrowd", 0):
                by_image[ann["image_id"]].append(ann)
        print(f"[prepare] Writing labels for {len(by_image):,} images ...", flush=True)
        paths, kept, skipped = [], 0, 0
        for img_id, anns in tqdm(by_image.items(), desc=ann_file.stem):
            info = img_info[img_id]
            W, H = info["width"], info["height"]
            img_path = img_dir / info["file_name"]
            if not img_path.exists():
                skipped += 1; continue
            lines = []
            for ann in anns:
                cls = COCO_KEEP[ann["category_id"]]
                x, y, w, h = ann["bbox"]
                cx = max(0.0, min(1.0, (x + w/2) / W))
                cy = max(0.0, min(1.0, (y + h/2) / H))
                nw = max(0.0, min(1.0, w / W))
                nh = max(0.0, min(1.0, h / H))
                if nw > 0 and nh > 0:
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            if lines:
                (label_dir / Path(info["file_name"]).stem).with_suffix(".txt").write_text("\n".join(lines))
                paths.append(str(img_path)); kept += 1
            else:
                skipped += 1
        list_out.write_text("\n".join(paths))
        print(f"  {ann_file.stem}: {kept:,} kept, {skipped:,} skipped")

    ann_dir = COCO_ROOT / "annotations"
    print("[prepare] Starting train split ...", flush=True)
    build(ann_dir / "instances_train2017.json", COCO_ROOT / "train2017", labels_root / "train2017", ROOT / "data" / "train.txt")
    print("[prepare] Starting val split ...", flush=True)
    build(ann_dir / "instances_val2017.json",   COCO_ROOT / "val2017",   labels_root / "val2017",   ROOT / "data" / "val.txt")

    names = "\n".join(f"  {i}: {n}" for i, n in CLASS_NAMES.items())
    yaml_path.write_text(f"path: {ROOT}\ntrain: data/train.txt\nval: data/val.txt\nnc: 2\nnames:\n{names}")
    print(f"YAML → {yaml_path}")


def train(size="n", epochs=50, batch=8, imgsz=1280):
    model_name = f"yolo26{size}"
    YOLO(f"{model_name}.pt").train(
        data          = str(ROOT / "data" / "football.yaml"),
        project       = str(ROOT / "runs"),
        name          = f"{model_name}_football",
        epochs        = epochs,
        imgsz         = imgsz,
        batch         = batch,
        device        = get_device(),
        rect          = True,
        workers       = 4,
        patience      = 20,
        lr0           = 0.01,
        lrf           = 0.01,
        warmup_epochs = 3,
        mosaic        = 1.0,
        mixup         = 0.1,
        copy_paste    = 0.1,
        fliplr        = 0.5,
        degrees       = 5.0,
        cache         = "disk",
        exist_ok      = True,
        plots         = True,
    )


def validate(size="n", imgsz=640):
    w = best_weights(size)
    print(f"[validate] Loading model {w.name} ...", flush=True)
    metrics = YOLO(str(w)).val(data=str(ROOT / "data" / "football.yaml"), imgsz=imgsz, device=get_device())
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    for i, ap in enumerate(metrics.box.ap50):
        print(f"  {CLASS_NAMES[i]}: {ap:.4f}")


def export(size="n", imgsz=640):
    w = best_weights(size)
    print(f"[export] Loading model {w.name} ...", flush=True)
    model = YOLO(str(w))
    print("[export] Exporting CoreML ...", flush=True)
    model.export(format="coreml", imgsz=imgsz, nms=True,  half=False)
    print("[export] Exporting ONNX ...", flush=True)
    model.export(format="onnx",   imgsz=imgsz, opset=17,  simplify=True)
    print("[export] Done.", flush=True)


def track(video_in, size="n", conf=0.25, imgsz=1280):
    video_in = Path(video_in)
    video_out = ROOT / "runs" / f"{video_in.stem}_tracked.mp4"
    print(f"[track] Loading model ...", flush=True)
    YOLO(str(best_weights(size))).track(
        source=str(video_in), tracker="bytetrack.yaml",
        conf=conf, imgsz=imgsz, device=get_device(),
        persist=True, save=True,
        project=str(ROOT / "runs"), name="tracking", exist_ok=True,
    )
    tracked = glob.glob(str(ROOT / "runs" / "tracking" / "*.mp4"))
    if tracked:
        shutil.copy(tracked[0], video_out)
        print(f"Saved → {video_out}")


def extract_csv(video_in, size="n", conf=0.25, imgsz=1280):
    video_in = Path(video_in)
    csv_out  = ROOT / "runs" / f"{video_in.stem}_detections.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[csv] Loading model ...", flush=True)
    model = YOLO(str(best_weights(size)))
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "cx", "cy"])
        for i, r in enumerate(model.track(source=str(video_in), tracker="bytetrack.yaml",
                                           conf=conf, imgsz=imgsz, device=get_device(),
                                           persist=True, stream=True, verbose=False)):
            if not r.boxes: continue
            for box in r.boxes:
                cls = int(box.cls.item())
                x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
                w.writerow([i, int(box.id.item()) if box.id is not None else -1,
                             CLASS_NAMES[cls], round(float(box.conf.item()), 4),
                             x1, y1, x2, y2, round((x1+x2)/2, 1), round((y1+y2)/2, 1)])
            if i % 100 == 0: print(f"  frame {i:,}")
    print(f"Saved → {csv_out}")


def label(images_dir, size="n", conf=0.25, imgsz=1280):
    images_dir = Path(images_dir)
    out = ROOT / "data" / "labeled"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    print(f"[label] Loading model ...", flush=True)
    model = YOLO(str(best_weights(size)))
    print(f"[label] Running predictions on {images_dir} ...", flush=True)
    saved = empty = 0
    for r in model.predict(source=str(images_dir), conf=conf, imgsz=imgsz,
                            device=get_device(), stream=True):
        p = Path(r.path)
        shutil.copy(p, out / "images" / p.name)
        if not r.boxes or len(r.boxes) == 0:
            empty += 1; continue
        lines = [f"{int(b.cls.item())} {' '.join(f'{v:.6f}' for v in b.xywhn[0].tolist())}"
                 for b in r.boxes]
        (out / "labels" / p.stem).with_suffix(".txt").write_text("\n".join(lines))
        saved += 1
    print(f"Labeled: {saved}  |  Empty: {empty}  →  {out}")


def stats(csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    total_frames  = df["frame"].nunique()
    frames_w_ball = int((df["class"] == "sports_ball").groupby(df["frame"]).any().sum())

    print(f"\nDetections  : {len(df):,}")
    print(f"Frames      : {total_frames:,}")
    print(f"Mean conf   : {df['conf'].mean():.4f}")
    print(f"Track IDs   : {df['track_id'].nunique()}")
    print(f"Ball in frame: {frames_w_ball/total_frames*100:.1f}%")
    for cls, n in df["class"].value_counts().items():
        print(f"  {cls}: {n:,}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, name in zip(axes[:2], CLASS_NAMES.values()):
        ax.hist(df[df["class"] == name]["conf"], bins=20, edgecolor="white")
        ax.set_title(f"{name} confidence"); ax.set_xlim(0, 1)
    per_frame = df.groupby(["frame", "class"]).size().unstack(fill_value=0)
    for col in per_frame.columns:
        axes[2].plot(per_frame.index, per_frame[col], label=col, linewidth=0.8)
    axes[2].set_title("Detections per frame"); axes[2].legend()
    plt.tight_layout(); plt.savefig(Path(csv_path).parent / "stats.png", dpi=150)
    print(f"Plot saved → {Path(csv_path).parent / 'stats.png'}")


def collect_used_images():
    dest = ROOT / "data" / "coco_filtered"
    for split in ("train2017", "val2017"):
        (dest / split).mkdir(parents=True, exist_ok=True)

    for txt, split in [(ROOT / "data/train.txt", "train2017"),
                       (ROOT / "data/val.txt",   "val2017")]:
        if not txt.exists():
            print(f"[collect] {txt.name} not found — run prepare first.")
            continue
        paths = [p for p in txt.read_text().splitlines() if p]
        print(f"[collect] Copying {len(paths):,} {split} images ...", flush=True)
        for p in tqdm(paths, desc=split, unit="img"):
            shutil.copy(p, dest / split / Path(p).name)

    print(f"[collect] Done → {dest}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare")

    t = sub.add_parser("train")
    t.add_argument("--size",   default="n", choices=["n","s","m","l"])
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch",  type=int, default=8)
    t.add_argument("--imgsz",  type=int, default=1280)

    v = sub.add_parser("validate")
    v.add_argument("--size", default="n", choices=["n","s","m","l"])

    e = sub.add_parser("export")
    e.add_argument("--size",  default="n", choices=["n","s","m","l"])
    e.add_argument("--imgsz", type=int, default=640)

    tk = sub.add_parser("track")
    tk.add_argument("--video", required=True)
    tk.add_argument("--size",  default="n", choices=["n","s","m","l"])
    tk.add_argument("--conf",  type=float, default=0.25)

    c = sub.add_parser("csv")
    c.add_argument("--video", required=True)
    c.add_argument("--size",  default="n", choices=["n","s","m","l"])
    c.add_argument("--conf",  type=float, default=0.25)

    lb = sub.add_parser("label")
    lb.add_argument("--images", required=True)
    lb.add_argument("--size",   default="n", choices=["n","s","m","l"])
    lb.add_argument("--conf",   type=float, default=0.25)

    sub.add_parser("collect")

    st = sub.add_parser("stats")
    st.add_argument("--csv", required=True)

    args = p.parse_args()

    if   args.cmd == "prepare":  prepare()
    elif args.cmd == "train":    train(args.size, args.epochs, args.batch, args.imgsz)
    elif args.cmd == "validate": validate(args.size)
    elif args.cmd == "export":   export(args.size, args.imgsz)
    elif args.cmd == "track":    track(args.video, args.size, args.conf)
    elif args.cmd == "csv":      extract_csv(args.video, args.size, args.conf)
    elif args.cmd == "label":    label(args.images, args.size, args.conf)
    elif args.cmd == "stats":    stats(args.csv)
    elif args.cmd == "collect":  collect_used_images()