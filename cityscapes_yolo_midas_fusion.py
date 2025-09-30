#!/usr/bin/env python3
"""
Cityscapes Obstacle Recognition with Spatial Resolution (YOLOv8 + MiDaS)
-------------------------------------------------------------------------
Beginner-friendly end-to-end pipeline:
1) Load a Cityscapes RGB image
2) Detect obstacles with YOLOv8 (pretrained on COCO)
3) Predict monocular depth with MiDaS (DPT-Large)
4) Fuse detection + depth to compute per-object distance (median depth) and direction
5) Visualize and export results (annotated image + CSV)

Environment (recommended):
- Python 3.9+
- PyTorch (with CUDA if available)
- ultralytics (YOLOv8)
- timm (MiDaS dependency)
- OpenCV, numpy, matplotlib

Install (example):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # (pick the right CUDA/CPU wheel)
    pip install ultralytics opencv-python matplotlib timm

Notes:
- MiDaS is loaded via torch.hub (weights auto-downloaded on first run).
- For Cityscapes, point IMG_PATH to any RGB image from leftImg8bit (e.g., .../leftImg8bit/val/frankfurt/xxx_leftImg8bit.png).
- Distances are *relative depth* values. Optional simple calibration to approximate meters is provided.

Author: (Your Name)
"""
import os
import sys
import math
import csv
import time
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------
# Config
# ---------------------------------------------
# Path to a single Cityscapes RGB image for a smoke test:
IMG_PATH = os.environ.get("CITYSCAPES_IMG", "path/to/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png")

# Output paths
OUT_DIR = os.environ.get("OUT_DIR", "./outputs")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_IMAGE = os.path.join(OUT_DIR, "annotated.png")
OUT_CSV   = os.path.join(OUT_DIR, "detections_with_distance.csv")

# Detection model (YOLOv8 weights)
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")  # try yolov8m.pt or yolov8l.pt for better accuracy

# MiDaS model spec
MIDAS_MODEL_NAME = os.environ.get("MIDAS_MODEL", "DPT_Large")  # "DPT_Large" (best), "DPT_Hybrid" (faster)

# Classes we consider obstacles (COCO indices for YOLOv8)
# COCO class names from Ultralytics (80 classes). We'll filter relevant obstacles for outdoor scenes.
OBSTACLE_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "stop sign", "bench"  # add more if desired
}
# ---------------------------------------------


def load_image(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img_bgr


def run_yolo(img_bgr: np.ndarray, weights: str = YOLO_WEIGHTS, conf: float = 0.25):
    """
    Run YOLOv8 inference on a BGR image.
    Returns a list of detections: dict with keys [cls_name, conf, xyxy(np.array)]
    """
    model = YOLO(weights)
    # Ultralytics expects path or array in RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, conf=conf, verbose=False)[0]

    detections = []
    names = model.model.names  # class index -> name
    if results.boxes is not None and len(results.boxes) > 0:
        for b in results.boxes:
            cls_id = int(b.cls.item())
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name not in OBSTACLE_CLASS_NAMES:
                continue
            xyxy = b.xyxy.squeeze().cpu().numpy()  # [x1, y1, x2, y2]
            conf_score = float(b.conf.item())
            detections.append({
                "cls_name": cls_name,
                "conf": conf_score,
                "xyxy": xyxy
            })
    return detections


def load_midas(model_name: str = MIDAS_MODEL_NAME):
    """
    Load MiDaS model + transform via torch.hub.
    Returns: (model, transform, is_cuda)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas = torch.hub.load("isl-org/MiDaS", model_name)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("isl-org/MiDaS", "transforms")
    if "DPT" in model_name:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return midas, transform, device


def run_midas_depth(img_bgr: np.ndarray, midas, transform, device: str) -> np.ndarray:
    """
    Compute relative depth map with MiDaS. Returns a float32 HxW array (larger = farther or nearer depending on model).
    We'll invert later if needed for intuition.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        depth = prediction.squeeze().cpu().numpy().astype(np.float32)

    # Normalize depth for visualization & consistency
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)  # 0..1
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    return depth_norm  # relative depth in [0,1] (higher means farther given our normalization)


def median_depth_in_box(depth_map: np.ndarray, xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = depth_map.shape[:2]
    x1 = np.clip(x1, 0, w-1)
    x2 = np.clip(x2, 0, w-1)
    y1 = np.clip(y1, 0, h-1)
    y2 = np.clip(y2, 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return float("nan")
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return float("nan")
    return float(np.median(roi))


def direction_from_box(xyxy: np.ndarray, img_w: int) -> str:
    x1, y1, x2, y2 = xyxy
    cx = 0.5 * (x1 + x2)
    left_thr = img_w / 3.0
    right_thr = 2.0 * img_w / 3.0
    if cx < left_thr:
        return "Left"
    elif cx > right_thr:
        return "Right"
    else:
        return "Center"


def relative_to_meters(depth_rel: float, a: float = 20.0, b: float = 1.0) -> float:
    """
    Optional naive conversion from relative depth [0,1] to 'meters' using linear scaling.
    Tune (a, b) on a small calibration set (e.g., a * depth_rel + b).
    Defaults produce arbitrary but monotonic output.
    """
    return float(a * depth_rel + b)


def annotate_and_save(img_bgr: np.ndarray, detections: List[Dict], depth_map: np.ndarray,
                      out_image_path: str, out_csv_path: str, convert_to_m: bool = False):
    os.makedirs(os.path.dirname(out_image_path), exist_ok=True)

    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()

    # Prepare CSV
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "confidence", "distance_rel", "distance_m_est", "direction", "x1", "y1", "x2", "y2"])

        for det in detections:
            cls_name = det["cls_name"]
            conf = det["conf"]
            xyxy = det["xyxy"]
            dist_rel = median_depth_in_box(depth_map, xyxy)  # 0..1 (farther = larger with our normalization)
            # Intuition: Invert so "closer" => larger number if you prefer. We'll keep as-is and label with meters.
            direction = direction_from_box(xyxy, w)
            if convert_to_m:
                dist_m = relative_to_meters(dist_rel)
            else:
                dist_m = None

            # Draw
            x1, y1, x2, y2 = xyxy.astype(int)
            color = (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f} | {'~'+str(dist_m)[:5]+' m' if dist_m is not None else f'rel={dist_rel:.2f}'} | {direction}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            writer.writerow([cls_name, f"{conf:.4f}", f"{dist_rel:.4f}", f"{dist_m:.4f}" if dist_m is not None else "",
                             direction, x1, y1, x2, y2])

    cv2.imwrite(out_image_path, vis)


def main():
    t0 = time.time()
    print("[INFO] Loading image:", IMG_PATH)
    img_bgr = load_image(IMG_PATH)

    print("[INFO] Running YOLOv8...")
    detections = run_yolo(img_bgr, weights=YOLO_WEIGHTS, conf=0.25)
    print(f"[INFO] Detections kept (obstacles only): {len(detections)}")

    print("[INFO] Loading MiDaS:", MIDAS_MODEL_NAME)
    midas, transform, device = load_midas(MIDAS_MODEL_NAME)

    print("[INFO] Predicting depth...")
    depth_rel = run_midas_depth(img_bgr, midas, transform, device)  # 0..1

    print("[INFO] Fusing detection + depth and saving outputs...")
    annotate_and_save(img_bgr, detections, depth_rel, OUT_IMAGE, OUT_CSV, convert_to_m=False)

    dt = time.time() - t0
    print(f"[DONE] Saved: {OUT_IMAGE}")
    print(f"[DONE] Saved: {OUT_CSV}")
    print(f"[TIME] {dt:.2f} s")


if __name__ == "__main__":
    if IMG_PATH == "path/to/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png":
        print("[WARNING] Please set IMG_PATH at the top of the script to a real Cityscapes image path.")
    main()
