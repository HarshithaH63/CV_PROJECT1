#!/usr/bin/env python3
"""
cityscapes_yolo_midas_fusion.py

Monocular obstacle detection + depth estimation fusion pipeline
using YOLOv8 and Depth-Anything (from Final.ipynb)
"""

import os
import time
import csv
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# -------------------- 1. Configuration --------------------
IMG_PATH = "./leftImg8bit_trainvaltest/val/frankfurt/frankfurt_000000_003025_leftImg8bit.png"
GT_DEPTH_PATH = "./gt_depth/frankfurt_000000_003025_depth.png"  # optional
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_IMAGE = os.path.join(OUT_DIR, "annotated.png")
OUT_CSV   = os.path.join(OUT_DIR, "detections_with_distance.csv")

YOLO_WEIGHTS = "yolov8n.pt"
OBSTACLE_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "bus",
    "truck", "train", "traffic light", "stop sign", "bench"
}

# Camera assumptions for approximate distance (optional)
IMAGE_WIDTH = 2048
ASSUMED_HORIZONTAL_FOV_DEG = 90.0
FOCAL_LENGTH_PX = IMAGE_WIDTH / (2 * np.tan(np.deg2rad(ASSUMED_HORIZONTAL_FOV_DEG / 2)))
AVG_HEIGHTS = {
    "person": 1.7, "car": 1.5, "bus": 3.0, "truck": 3.5,
    "bicycle": 1.2, "motorcycle": 1.2, "traffic light": 3.0, "bench": 0.8
}

# -------------------- 2. Helper Functions --------------------
def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.imread(path, cv2.IMREAD_COLOR)

def run_yolo(img_bgr, weights=YOLO_WEIGHTS, conf=0.25):
    model = YOLO(weights)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, conf=conf, verbose=False)[0]
    detections = []
    names = model.model.names
    for b in results.boxes:
        cls_id = int(b.cls.item())
        cls_name = names.get(cls_id, str(cls_id))
        if cls_name not in OBSTACLE_CLASS_NAMES: continue
        xyxy = b.xyxy.squeeze().cpu().numpy()
        conf_score = float(b.conf.item())
        detections.append({"cls_name": cls_name, "conf": conf_score, "xyxy": xyxy})
    return detections

def load_depth_anything(model_id="depth-anything/Depth-Anything-V2-small-hf"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device

def run_depth_anything(img_bgr, processor, model, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
    return (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min() + 1e-8)

def median_depth_in_box(depth_map, xyxy):
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = depth_map.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, w - 1)
    y1, y2 = np.clip([y1, y2], 0, h - 1)
    if x2 <= x1 or y2 <= y1: return float("nan")
    roi = depth_map[y1:y2, x1:x2]
    return float(np.median(roi)) if roi.size > 0 else float("nan")

def direction_from_box(xyxy, img_w):
    cx = 0.5 * (xyxy[0] + xyxy[2])
    if cx < img_w / 3: return "Left"
    elif cx > 2 * img_w / 3: return "Right"
    return "Center"

def approximate_distance(xyxy, cls_name, focal_px=FOCAL_LENGTH_PX):
    x1, y1, x2, y2 = xyxy.astype(int)
    bbox_height_px = max(1, y2 - y1)
    real_height_m = AVG_HEIGHTS.get(cls_name, 1.7)
    return (focal_px * real_height_m) / bbox_height_px

def annotate_and_save(img_bgr, detections, depth_map, out_image_path, out_csv_path):
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class","confidence","distance_rel","distance_m","direction","x1","y1","x2","y2"])
        detections_sorted = sorted(detections, key=lambda d: median_depth_in_box(depth_map, d["xyxy"]))
        for det in detections_sorted:
            dist_rel = median_depth_in_box(depth_map, det["xyxy"])
            direction = direction_from_box(det["xyxy"], w)
            x1, y1, x2, y2 = det["xyxy"].astype(int)
            dist_m = approximate_distance(det["xyxy"], det["cls_name"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['cls_name']} {det['conf']:.2f} | rel={dist_rel:.2f} | {dist_m:.1f}m | {direction}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0,y1-th-6)), (x1+tw+6, y1), (255,255,255), -1)
            cv2.putText(vis, label, (x1+3,max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            writer.writerow([det["cls_name"], f"{det['conf']:.4f}", f"{dist_rel:.4f}", f"{dist_m:.2f}", direction, x1, y1, x2, y2])
    cv2.imwrite(out_image_path, vis)
    return vis

def evaluate_depth(pred_depth, gt_path):
    if not os.path.exists(gt_path):
        print("[INFO] GT depth not found. Skipping depth evaluation.")
        return None
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    gt /= gt.max()
    rmse = np.sqrt(np.mean((pred_depth - gt)**2))
    mae = np.mean(np.abs(pred_depth - gt))
    print(f"[METRIC] Depth RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

def generate_observations(csv_path, depth_eval=None):
    df = pd.read_csv(csv_path)
    print("\n--- Quantitative Analysis ---")
    print("Total objects detected:", len(df))
    print("Detections per class:\n", df["class"].value_counts())
    print("Mean relative depth:", df["distance_rel"].mean())
    print("Closest obstacle:\n", df.loc[df["distance_rel"].idxmin()])
    print("Farthest obstacle:\n", df.loc[df["distance_rel"].idxmax()])
    print("Direction distribution:\n", df["direction"].value_counts())
    if depth_eval is not None:
        rmse, mae = depth_eval
        print(f"Optional depth evaluation -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Optional plots
    plt.figure(figsize=(8,5))
    plt.scatter(df["confidence"], df["distance_rel"], alpha=0.7)
    plt.xlabel("YOLO Confidence")
    plt.ylabel("Relative Depth (0=near)")
    plt.title("Confidence vs Relative Depth")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,5))
    df.groupby("class")["distance_rel"].mean().sort_values().plot(kind="bar", color="skyblue")
    plt.ylabel("Mean Relative Depth")
    plt.title("Average Distance by Object Class")
    plt.tight_layout()
    plt.show()

# -------------------- 3. Main Pipeline --------------------
def main():
    print("[INFO] Running YOLO + DepthAnything pipeline...")
    start_time = time.time()

    img_bgr = load_image(IMG_PATH)
    detections = run_yolo(img_bgr)
    print(f"[INFO] {len(detections)} detections found.")

    processor, depth_model, device = load_depth_anything()
    depth_rel = run_depth_anything(img_bgr, processor, depth_model, device)

    annotate_and_save(img_bgr, detections, depth_rel, OUT_IMAGE, OUT_CSV)
    print(f"[INFO] Annotated image saved to: {OUT_IMAGE}")
    print(f"[INFO] CSV saved to: {OUT_CSV}")

    depth_eval = evaluate_depth(depth_rel, GT_DEPTH_PATH)
    generate_observations(OUT_CSV, depth_eval)

    print(f"[INFO] Total pipeline time: {time.time() - start_time:.2f} seconds")

# -------------------- 4. Script Entry --------------------
if __name__ == "__main__":
    main()

