
# Simple Pilot: Lane Detection & Recording System

This project provides a complete solution for real-time lane detection, vehicle detection (ADAS), and video recording using a Logitech C920 webcam. It is optimized for Intel CPUs/iGPUs using OpenVINO.

## 🚀 Key Features

### 1. Robust Video Recording (c920_lane_detect.py)
- **High Performance Recording**: Threads capture and writing separately to ensure smooth 30 FPS recording without frame drops.
- **Auto-FPS Calibration**: Automatically detects camera's actual output FPS and adjusts recording speed to prevent "fast-forward" effect.
- **180° Flip**: Corrects orientation for mounted cameras.
- **H.264 Encoding**: Uses efficient compression (with fallback to MP4V if needed).
- **Auto-Timestamping**: Saves files as `output_YYYYMMDD_HHMMSS.mp4` to prevent overwrites.

### 2. Intelligent Lane Tracking (simple_pilot.py)
- **Lane Smoothing**: Implements a `LaneTracker` with history buffer to eliminate detection jitter.
- **Polynomial Fitting**: Draws smooth, continuous lane markers instead of jagged lines.
- **Robust Detection**: Combines HLS color filtering (Yellow/White) with Canny edge detection for maximum reliability.
- **Intel OpenVINO Acceleration**: Utilizes Intel iGPU for accelerating YOLOv8 detection, achieving 20+ FPS on standard laptops.

## 🛠️ Installation

```bash
pip install opencv-python numpy ultralytics openvino-dev
```

**Intel Optimization (Optional but Recommended):**
```bash
# Export YOLO model to OpenVINO format for 2x speedup
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='openvino')"
```

## 📖 Usage

### Recording & Basic Lane Detection
```bash
# Auto-detect FPS and start recording
python c920_lane_detect.py

# Specify camera index manually
python c920_lane_detect.py --camera 1
```

### Advanced ADAS (Lane + Vehicle Detection on Video)
```bash
# Run advanced pilot on recorded video
python simple_pilot.py
```

## 🔧 Technical Details
- **Resolution**: 1280x720 (720p)
- **Queued Buffering**: A thread-safe queue buffers frames between capture and disk writing to handle I/O spikes.
- **HLS Color Space**: Used to reliably detect lane markings under varying lighting conditions (especially yellow lines).
- **Stateful Tracking**: Uses `collections.deque` to maintain a moving average of lane coefficients.

## 📄 License
MIT License
