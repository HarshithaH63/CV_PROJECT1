# MonoDepth-Assist: Monocular Detection + Depth Fusion for Navigation

MonoDepth-Assist is a monocular object detection and depth estimation pipeline designed to enhance navigation for visually impaired users. By combining **YOLOv8** for real-time object detection with **Depth-Anything** for high-quality depth estimation, the system detects obstacles, estimates their relative distance, and provides intuitive Left/Center/Right spatial labeling from a single RGB image.

---

## Features

- **Real-time object detection** using YOLOv8
- **Monocular depth estimation** with Depth-Anything
- **Detection-depth fusion**: median depth per object, direction labeling
- Annotated image output + CSV with detection and depth info
- Optional evaluation against ground truth depth
- Lightweight, no extra sensors required

---

## Installation

Recommended: Python 3.9+ with PyTorch and CUDA (optional)

```bash
# Install PyTorch (pick the correct CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install ultralytics opencv-python matplotlib pandas transformers timm

bash ```

----
## Dataset
This project uses the Cityscapes dataset:

RGB images: leftImg8bit_trainvaltest/

Optional ground truth depth: gtFine_trainvaltest/ or custom depth maps

Note: The dataset is large (~10s of GBs), so these folders are ignored in Git via .gitignore.

Usage
Set the image path in cityscapes_yolo_midas_fusion.py:
IMG_PATH = "./leftImg8bit_trainvaltest/val/frankfurt/frankfurt_000000_003025_leftImg8bit.png"
GT_DEPTH_PATH = "./gt_depth/frankfurt_000000_003025_depth.png"  # optional
Run the pipeline: python cityscapes_yolo_midas_fusion.py
Outputs:
Annotated image: outputs/annotated.png
CSV with detections & depth: outputs/detections_with_distance.csv
Optional plots and summary statistics in the terminal or displayed via matplotlib

Example Output
Class	Confidence	Rel. Depth	Est. Distance (m)	Direction
car   	0.95	     0.32	       35.1	             Center
person	0.87	     0.45	       29.4	             Left

Annotated images will have bounding boxes, distance labels, and direction overlayed.

Optional Configuration
YOLO weights: yolov8n.pt, yolov8m.pt, or yolov8l.pt
Depth model: "depth-anything/Depth-Anything-V2-small-hf" (default, can change to larger model for higher quality)
Output directory: OUT_DIR in the script

License & References
YOLOv8: https://github.com/ultralytics/ultralytics
Depth-Anything: https://huggingface.co/depth-anything/Depth-Anything-V2-small-hf
Cityscapes dataset: https://www.cityscapes-dataset.com/
