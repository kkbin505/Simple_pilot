
# Simple Pilot Pro: ADAS & Lane Detection System

This project is an Advanced Driver Assistance System (ADAS) with two complementary approaches to real-time lane detection and vehicle safety, optimized for the Logitech C920 camera on Intel CPUs/iGPUs (Iris Xe).

---

## 🆕 Branch C920: Lightweight Traditional Lane Detection

**`c920_lane_detect.py`** — A new zero-dependency lane detector using classical computer vision, no deep learning required:

- ✅ **HLS Color Filtering**: Separate white and yellow lane line masks (robust in varied lighting)
- ✅ **Canny + Hough Transform**: Edge detection pipeline with ROI masking for reliable line extraction
- ✅ **Auto FPS Calibration**: Measures real camera FPS before recording to prevent video desync
- ✅ **Frame-Skip Optimization**: Runs heavy processing every 3rd frame, displays cached results at full speed
- ✅ **Multi-threaded I/O**: Dedicated capture and H.264 writer threads for zero frame drops
- ✅ **Auto Codec Detection**: Falls back from `avc1` to `mp4v` if OpenH264 DLL is not found

### Quick Start (Traditional, No ML)
```bash
# Run on C920 (default camera index 1)
python c920_lane_detect.py

# Specify camera index or override FPS
python c920_lane_detect.py --camera 0 --fps 30
```

| Argument | Default | Description |
|---|---|---|
| `--camera` | `1` | Camera device index |
| `--fps` | auto-detect | Recording FPS (measured from camera if not set) |

---

## 🚀 AI-Powered ADAS: `simple_pilot.py`

The full AI pipeline using YOLOv8-Seg + OpenVINO for deep learning lane segmentation and vehicle detection:

### Key Features

#### 1. Neural Lane Segmentation
- **YOLOv8-Seg Integration**: Deep learning segmentation model for superior reliability in complex lighting
- **Kalman Filter Tracking**: Smooth lane tracking with state vector `[m, b, dm, db]` (Constant Velocity model)
- **Weighted Mask Fitting**: Better accuracy by weighting points near the bottom of the image

#### 2. Intelligent ADAS Logic
- **Smart Motion Detection**: Pauses inference when stationary (saves ~70% CPU power)
- **FCW (Forward Collision Warning)**: Detects vehicles inside your lane using lane boundary geometry
- **LDW (Lane Departure Warning)**: Lane center offset calculation with configurable pixel threshold
- **Intel Iris Xe Acceleration**: OpenVINO FP16 inference on integrated GPU (2x speedup over CPU FP32)

#### 3. Vehicle Detection (Multi-Class)
- Detects cars, motorcycles, buses, and trucks (COCO classes 2, 3, 5, 7)
- Color-coded bounding boxes: 🟢 adjacent lane · 🟡 in-lane · 🔴 collision risk

#### 4. Professional Recording Engine
- Multi-threaded capture + H.264 writing
- C920 hardware focus lock (infinity) and auto-exposure management

---

## 🛠️ Installation

### Traditional Pipeline (`c920_lane_detect.py`)
```bash
pip install opencv-python numpy
```

### AI Pipeline (`simple_pilot.py`)
```bash
pip install opencv-python numpy ultralytics openvino-dev
```

### Export Model for Intel Iris Xe GPU
```bash
# Export with FP16 and 640px (recommended for full accuracy)
yolo export model=yolov8n-seg.pt format=openvino imgsz=640 half=True

# Or lightweight 320px for higher FPS
yolo export model=yolov8n-seg.pt format=openvino imgsz=320 half=True
```
> This creates a `yolov8n-seg_openvino_model/` folder. `simple_pilot.py` auto-detects and loads it.

---

## 📖 Usage

### Traditional Lane Detect (C920 branch)
```bash
python c920_lane_detect.py              # Auto FPS, camera index 1
python c920_lane_detect.py --camera 0   # Different camera
python c920_lane_detect.py --fps 15     # Manual FPS override
```

### AI Smart Pilot
```bash
python simple_pilot.py                  # Live camera
python simple_pilot.py video.mp4        # Run on pre-recorded file
python simple_pilot.py video.mp4 --start_sec 30  # Skip first 30s
```

### Controls (both scripts)
| Key | Action |
|---|---|
| **`q`** | Safe shutdown — saves video and releases camera |

---

## 🔧 Calibration

Tune these constants in the `# Configuration` section of each script:

| Parameter | Script | Description |
|---|---|---|
| `LDW_THRESHOLD_PX` | `simple_pilot.py` | Pixel offset from center to trigger LDW |
| `INFERENCE_SIZE` | `simple_pilot.py` | Must match your exported OpenVINO model size |
| `MOTION_CHANGE_RATIO` | `simple_pilot.py` | Sensitivity of stationary detection |
| ROI polygon | `c920_lane_detect.py` | `region_of_interest()` trapezoid vertices |
| HLS thresholds | `c920_lane_detect.py` | White/yellow color filter ranges |

---

## 📁 Project Structure

```
Simple_pilot/
├── simple_pilot.py            # AI ADAS (YOLOv8-Seg + Kalman + FCW/LDW)
├── simple_pilot_optimized.py  # Refactored AI version with extra optimizations
├── c920_lane_detect.py        # 🆕 Traditional CV lane detect (C920 branch)
├── export_openvino.py         # Model export helper
├── benchmark.py               # Performance benchmarking
├── yolov8n-seg.pt             # YOLOv8 segmentation model
├── yolov8n-seg_openvino_model/ # Exported OpenVINO model
├── requirement.txt            # Python dependencies
├── ARCHITECTURE.md            # System design documentation
├── OPTIMIZATION_GUIDE.md      # Performance tuning guide
└── calibration_data.npz       # Camera calibration data
```

---

## 📄 License
MIT License
