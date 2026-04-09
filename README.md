# Simple Pilot: Vision-Based Driving Assistant

Simple Pilot is a lightweight, high-performance Advanced Driver Assistance System (ADAS) that provides real-time Lane Departure Warning (LDW) and Forward Collision Warning (FCW) using a hybrid approach of classical computer vision and deep learning.

---

## 🚀 Key Improvements & Features

### 1. Advanced Lane Detection Pipeline
- **Sharpness-First Detection**: Performs edge detection on original pixels for maximum precision before any geometric warping.
- **CLAHE Contrast Enhancement**: Uses Contrast Limited Adaptive Histogram Equalization to restore visibility of faint lane markers in shadows or overexposed conditions.
- **HSV Dual-Color Masking**: Robust detection of both **White** and **Yellow** lanes using multi-channel color segmentation.
- **Triple-Guard Filtering**:
    - **Top 35% Exclusion**: Ignores sky, horizon, and trees to prevent false positives.
    - **Bottom 10% Exclusion**: Clips the car hood and dashboard reflections.
    - **0.5 Slope Filter**: Aggressively discards horizontal road textures and artifacts.

### 2. Temporal Stability & Tracking
- **LaneTracker Engine**: Replaced basic LPF with a **deque-based historical smoothing (SMA)**.
- **Outlier Rejection**: A sliding window of 10 frames eliminates artifacts and keeps the lane stable during bumpy conditions or momentary dropouts.

### 3. Lens Calibration (Optional)
- **Fisheye Correction**: Support for OpenCV/Gyroflow lens profile JSON files to rectify wide-angle distortion for geometric accuracy.
- **Correction-After-Detection**: Hybrid workflow that detects on sharp "bent" pixels but calculates warnings on "straightened" coordinates.

### 5. Deep Learning Vehicle Detection (YOLOv8)
- **Object Recognition**: Identifies cars, trucks, and buses in real-time.
- **FCW Logic**: Collision warning based on object proximity and relative position within the lane.

---

## 🛠️ Installation

### Dependencies
```bash
pip install opencv-python numpy torch ultralytics
```

---

## 📖 Usage

### Running the Script
Run on a video file or a live camera stream:

```bash
# Basic run on a video file
python simple_pilot.py path/to/driving_video.mp4

# Advanced: Start from 100 seconds into the video
python simple_pilot.py "D:\video.mp4" -s 100

# Run on live webcam (index 0)
python simple_pilot.py 0
```

### Controls
| Key | Action |
|---|---|
| **`q`** | Safe shutdown and release resources |

---

## 📁 Project Structure

```text
Simple_pilot/
├── simple_pilot.py      # Core ADAS Logic (Lane Detection + FCW + LDW)
├── yolov8n.pt           # YOLOv8 pre-trained weights
├── lens/                # (Optional) Lens calibration JSON files
└── README.md            # Project documentation
```

---

## 📄 License
This project is released under the MIT License.
