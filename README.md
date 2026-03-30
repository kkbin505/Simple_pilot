# Simple Pilot: Vision-Based Driving Assistant

Simple Pilot is a lightweight Advanced Driver Assistance System (ADAS) demo that provides real-time Lane Departure Warning (LDW) and Forward Collision Warning (FCW) using computer vision and deep learning.

---

## 🚀 Recent Updates & Features

### 1. Robust Lane Detection with LPF
- **Temporal Smoothing (Low Pass Filter)**: Uses an exponential moving average on lane line coefficients to eliminate flickering and provide stable visualization.
- **Optimized & Dynamic ROI**: Features an adjustable trapezoidal mask with configurable offsets and extension parameters, ensuring precise road focus and minimal background noise.
- **Temporal Smoothing (LPF)**: Uses an exponential moving average on lane line coefficients to eliminate flickering and provide stable visualization.
- **Canny + Hough Pipeline**: Efficient classical CV pipeline for lane extraction without heavy GPU requirements.

### 2. Deep Learning Vehicle Detection (YOLOv8)
- **Object Recognition**: Identifies cars, trucks, buses, and motorcycles in real-time.
- **FCW Logic**: Simple but effective collision warning based on bounding box size and lane position.
- **Hardware Acceleration**: Automatic CUDA detection for NVIDIA GPUs.

### 3. Smart Command Line Interface
- **Flexible Source Input**: Supports both local video files and live camera streams (via index).
- **Time Seeking**: Jump to any specific second in a video using the `-s` flag.
- **Error Handling**: Built-in fixes for common OpenMP (`OMP Error #15`) and environment initialization issues.

---

## 🛠️ Installation

### Dependencies
```bash
pip install opencv-python numpy torch ultralytics
```

---

## 📖 Usage

### Running the Script
Provide a video path or a camera index (default is `0` for webcam):

```bash
# Run on a video file
python simple_pilot.py path/to/driving_video.mp4

# Run on a video starting from the 100th second
python simple_pilot.py "D:\video.mp4" -s 100

# Run on a specific connected camera (index 1)
python simple_pilot.py 1
```

### Controls
| Key | Action |
|---|---|
| **`q`** | Safe shutdown and release resources |

---

## 🔧 Configuration (Calibration)

Open `simple_pilot.py` to adjust these variables for better performance:
- `LPF_ALPHA`: Adjust smoothing (lower = smoother, higher = faster).
- `width_ratio` / `left_offset`: Calibrate the lane detection trapezoid.
- `minLineLength`: Threshold for Hough line detection.

---

## 📁 Project Structure

```text
Simple_pilot/
├── simple_pilot.py      # Main script (Lane Detect + FCW + LDW + LPF Filter)
├── yolov8n.pt           # YOLOv8 pre-trained weights
└── README.md            # Updated documentation
```

---

## 📄 License
This project is released under the MIT License.
