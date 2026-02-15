
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
- **Hybrid Mode (Camera/File)**: Supports both live C920 camera input and video file inference.
- **Live C920 Integration**:
    -   **Threaded Capture**: Ensures high FPS recording and inference.
    -   **Focus Locking**: Automatically locks focus to infinity (0) for clear road view.
    -   **180° Flip**: Handles camera mounting orientation.
    -   **Recording**: Saves live feed to `output_YYYYMMDD_HHMMSS.mp4`.
- **Lane Smoothing**: Implements a `LaneTracker` with history buffer to eliminate detection jitter.
- **Polynomial Fitting**: Draws smooth, continuous lane markers instead of jagged lines.
- **Robust Detection**: Combines HLS color filtering (Yellow/White) with Canny edge detection for maximum reliability.
- **Intel OpenVINO Acceleration**: Utilizes Intel iGPU for accelerating YOLOv8 detection, achieving 20+ FPS on standard laptops.
- **Screenshot Feature**: Press **'s'** to save the current frame with inference overlays as a high-quality JPG.

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

### Live Camera Mode (Default)
Run `simple_pilot.py` without arguments to use the C920 camera:
```bash
python simple_pilot.py
```
-   **Features**: Recording, Focus Lock, 180° Flip, Real-time Inference.
-   **Controls**:
    -   **'s'**: Save screenshot.
    -   **'q'**: Quit.

### Video File Mode
Run `simple_pilot.py` with a filename to process a video:
```bash
python simple_pilot.py input_video.mp4
```
-   **Features**: Inference on pre-recorded video (No recording/flip/focus lock).

## 🔧 Technical Details
- **Resolution**: 1280x720 (720p)
- **Queued Buffering**: A thread-safe queue buffers frames between capture and disk writing to handle I/O spikes.
- **HLS Color Space**: Used to reliably detect lane markings under varying lighting conditions (especially yellow lines).
- **Stateful Tracking**: Uses `collections.deque` to maintain a moving average of lane coefficients.

## 📄 License
MIT License
