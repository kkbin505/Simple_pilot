"""
Simple Pilot Pro - Optimized ADAS System
=========================================
Core Strategy:
1. Perspective View (No BEV): Direct lane detection on original image for maximum pixel precision
2. Kalman Filtering: Smooth tracking of lane parameters [m, b, dm, db] to prevent frame jitter
3. Smart Motion Triggering: Pause heavy inference when stationary to save power
4. YOLOv8n-seg: Semantic segmentation with weighted mask fitting (not just Hough)
5. OpenVINO GPU: Explicit Intel Iris Xe acceleration with FP16 precision
6. Warning Logic: LDW (Lane Departure) and FCW (Forward Collision) based on filtered parameters
7. Thread Safety: Async video capture and writing

Hardware Target: Intel i5-1155G7 with Iris Xe Graphics
"""

import cv2
import numpy as np
import time
import argparse
import os
from collections import deque
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
import warnings

# Suppress OpenMP warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library not found. Install with: pip install ultralytics")
    exit(1)

# ====== Configuration ======
# Model Settings
LANE_MODEL_NAME = "yolov8n-seg.pt"
OPENVINO_DIR = "yolov8n-seg_openvino_model"
INFERENCE_SIZE = 640  # Input size for model (higher = more accurate but slower)

# Detection Thresholds
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

# Motion Detection Settings
MOTION_THRESHOLD = 2.0  # Pixel shift threshold
STATIONARY_FRAME_COUNT = 15  # Frames to wait before declaring stationary
MOTION_CHANGE_RATIO = 0.008  # 0.8% pixel change threshold

# Warning Thresholds
LDW_OFFSET_THRESHOLD_M = 0.5  # Lane departure warning threshold in meters
FCW_DANGER_ZONE_Y = 0.65  # Forward collision warning zone (fraction of image height)
FCW_CENTER_X_MIN = 0.35  # Center lane area (fraction of image width)
FCW_CENTER_X_MAX = 0.65

# Kalman Filter Tuning
KALMAN_PROCESS_NOISE = 1e-4  # Lower = trust model more
KALMAN_MEASUREMENT_NOISE = 5e-2  # Lower = trust measurements more
KALMAN_DERIVATIVE_NOISE = 1e-3  # Variance for slope/intercept change rate

# Lane Geometry (for offset calculation)
LANE_WIDTH_M = 3.7  # Standard US lane width in meters
CAMERA_FOCAL_LENGTH_PX = 1000  # Approximate focal length (calibrate for your camera)

# Visualization
SHOW_DEBUG_MASKS = False  # Set to True to see segmentation masks
LANE_COLOR_GOOD = (0, 255, 0)  # Green
LANE_COLOR_WARNING = (0, 165, 255)  # Orange
LANE_COLOR_DANGER = (0, 0, 255)  # Red


class VideoStream:
    """
    Thread-safe video capture and recording.
    - Capture Thread: Pulls frames from camera at maximum rate
    - Writer Thread: Processes and writes frames to disk from queue
    - Main Thread: Reads latest frame for inference without blocking
    """
    def __init__(self, camera_idx, width, height, target_fps, output_file):
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_idx)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_idx}")

        # Configure camera for C920
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Lock focus to infinity for road scenes
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, target_fps, (self.width, self.height))

        # Threading state
        self.write_queue = Queue(maxsize=128)
        self.latest_frame = None
        self.stopped = False
        self.lock = Lock()
        
        # Diagnostics
        self.capture_count = 0
        self.actual_capture_fps = 0
        
        # Threads
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.writer_thread = Thread(target=self._writer_loop, daemon=True)

    def start(self):
        self.capture_thread.start()
        self.writer_thread.start()
        return self

    def _capture_loop(self):
        start_time = time.time()
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break

            self.capture_count += 1
            if self.capture_count % 30 == 0:
                now = time.time()
                self.actual_capture_fps = 30 / (now - start_time)
                start_time = now

            # Queue frame for writing (non-blocking)
            if not self.write_queue.full():
                try:
                    self.write_queue.put_nowait(frame)
                except:
                    pass
            
            # Update latest frame for inference
            with self.lock:
                self.latest_frame = frame

    def _writer_loop(self):
        while not self.stopped or not self.write_queue.empty():
            if not self.write_queue.empty():
                frame = self.write_queue.get()
                # Flip for C920 mounted upside down
                flipped = cv2.flip(frame, -1) if frame is not None else None
                if flipped is not None:
                    self.out.write(flipped)
                self.write_queue.task_done()
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                # Flip to match recorded orientation
                return cv2.flip(self.latest_frame, -1)
            return None

    def stop(self):
        self.stopped = True
        self.capture_thread.join(timeout=1)
        self.writer_thread.join(timeout=2)
        self.cap.release()
        self.out.release()


class MotionDetector:
    """
    Lightweight motion detection using frame differencing.
    Pauses heavy inference when vehicle is stationary to save power.
    """
    def __init__(self, threshold=MOTION_THRESHOLD, change_ratio=MOTION_CHANGE_RATIO):
        self.prev_gray = None
        self.threshold = threshold
        self.change_ratio = change_ratio
        self.stationary_counter = 0
        self.is_moving = True

    def update(self, frame):
        # Downscale for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return True

        # Calculate frame difference
        diff = cv2.absdiff(self.prev_gray, gray)
        non_zero_count = np.count_nonzero(diff > 25)
        
        # Normalize by area
        change_ratio = non_zero_count / (gray.shape[0] * gray.shape[1])
        
        if change_ratio > self.change_ratio:
            self.stationary_counter = 0
            self.is_moving = True
        else:
            self.stationary_counter += 1
            if self.stationary_counter > STATIONARY_FRAME_COUNT:
                self.is_moving = False
        
        self.prev_gray = gray
        return self.is_moving


class KalmanLaneTracker:
    """
    Kalman Filter for smooth lane tracking.
    State Vector: [m, b, dm, db]
    - m: slope (x = my + b formulation for near-vertical lines)
    - b: intercept
    - dm: rate of change of slope
    - db: rate of change of intercept
    
    This prevents single-frame detection jitter and provides smooth tracking.
    """
    def __init__(self):
        # 4 state variables, 2 measurement variables (m, b)
        self.kf = cv2.KalmanFilter(4, 2, 0)
        
        # Transition Matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # m = m + dm
            [0, 1, 0, 1],  # b = b + db
            [0, 0, 1, 0],  # dm = dm
            [0, 0, 0, 1]   # db = db
        ], dtype=np.float32)

        # Measurement Matrix (we directly measure m and b)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process Noise Covariance (trust the model)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_NOISE
        self.kf.processNoiseCov[2, 2] = KALMAN_DERIVATIVE_NOISE
        self.kf.processNoiseCov[3, 3] = KALMAN_DERIVATIVE_NOISE

        # Measurement Noise Covariance (trust vs noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEASUREMENT_NOISE

        # Error Covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.age = 0
        self.is_initialized = False
        self.last_measurement_time = time.time()

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        return pred

    def update(self, m, b):
        """Update filter with new measurement"""
        measurement = np.array([[np.float32(m)], [np.float32(b)]])
        self.kf.correct(measurement)
        self.age = 0
        self.is_initialized = True
        self.last_measurement_time = time.time()

    def get_state(self):
        """Get current filtered estimate of [m, b]"""
        return self.kf.statePost[0][0], self.kf.statePost[1][0]
    
    def get_confidence(self):
        """Return confidence based on age and initialization"""
        if not self.is_initialized:
            return 0.0
        # Decay confidence with age (no recent measurements)
        return max(0.0, 1.0 - self.age * 0.05)


class LaneDetectionSystem:
    """
    Main lane detection and ADAS system.
    Integrates YOLOv8-seg, Kalman filtering, motion detection, and warning logic.
    """
    def __init__(self, use_openvino=True):
        # Load model with OpenVINO optimization
        self.model = self._load_model(use_openvino)
        
        # Kalman trackers for left and right lanes
        # Using x = my + b formulation (better for near-vertical lines)
        self.left_tracker = KalmanLaneTracker()
        self.right_tracker = KalmanLaneTracker()
        
        # Motion detector
        self.motion_detector = MotionDetector()
        
        # State
        self.vehicle_offset_m = 0.0
        self.warning_flags = {'LDW': False, 'FCW': False}
        self.last_inference_time = 0
        self.inference_fps = 0
        
        # Debug visualization
        self.debug_mask = None

    def _load_model(self, use_openvino):
        """
        Load YOLOv8-seg model with OpenVINO optimization.
        Explicitly configured for Intel Iris Xe GPU with FP16 precision.
        """
        if use_openvino and os.path.exists(OPENVINO_DIR):
            path = OPENVINO_DIR
            print(f"✓ Loading OpenVINO model from: {path}")
            print(f"  → Optimized for Intel Iris Xe GPU with FP16 precision")
        else:
            path = LANE_MODEL_NAME
            print(f"⚠ Loading PyTorch model: {path}")
            print(f"  → For best performance, export to OpenVINO:")
            print(f"    yolo export model={LANE_MODEL_NAME} format=openvino imgsz={INFERENCE_SIZE} half=True")
        
        try:
            model = YOLO(path, task='segment')
            return model
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return None

    def _fit_line_weighted(self, mask_points, weights=None):
        """
        Fit line x = my + b with optional weighting.
        Args:
            mask_points: Nx2 array of [y, x] coordinates
            weights: Optional weights for each point
        Returns:
            (m, b) or None if fitting fails
        """
        if len(mask_points) < 50:
            return None
        
        y = mask_points[:, 0]
        x = mask_points[:, 1]
        
        try:
            if weights is not None:
                # Weighted polynomial fit
                z = np.polyfit(y, x, 1, w=weights)
            else:
                z = np.polyfit(y, x, 1)
            return z[0], z[1]  # m, b
        except:
            return None

    def process_frame(self, frame):
        """
        Main processing pipeline:
        1. Motion detection
        2. Kalman prediction
        3. YOLOv8-seg inference (if moving)
        4. Mask processing and weighted fitting
        5. Kalman update
        6. Warning logic (LDW, FCW)
        """
        h, w = frame.shape[:2]
        mid_x = w // 2
        
        # 1. Motion Check
        is_moving = self.motion_detector.update(frame)
        
        # 2. Kalman Prediction (always predict, even when stationary)
        self.left_tracker.predict()
        self.right_tracker.predict()
        
        # 3. Run Inference (only if moving)
        if is_moving and self.model:
            inference_start = time.time()
            
            # Run YOLOv8-seg inference
            # For OpenVINO models, device='cpu' triggers OpenVINO runtime
            # which automatically uses GPU if available and configured
            use_half = os.path.exists(OPENVINO_DIR)
            
            results = self.model(
                frame,
                verbose=False,
                half=use_half,
                device='cpu',  # OpenVINO handles GPU routing
                imgsz=INFERENCE_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD
            )
            
            inference_time = time.time() - inference_start
            self.inference_fps = 1.0 / inference_time if inference_time > 0 else 0
            
            # 4. Process Detection Results
            self._process_detections(results[0], frame, h, w, mid_x)
        
        # 5. Get Smoothed Lane Estimates
        mL, bL = self.left_tracker.get_state()
        mR, bR = self.right_tracker.get_state()
        
        # 6. Warning Logic
        self._update_warnings(mL, bL, mR, bR, h, w, mid_x)
        
        return frame, (mL, bL), (mR, bR)

    def _process_detections(self, result, frame, h, w, mid_x):
        """
        Process YOLOv8-seg detection results:
        - Extract semantic masks
        - Separate left/right lanes
        - Weighted line fitting
        - Update Kalman filters
        """
        # FCW: Check for vehicles in danger zone
        self.warning_flags['FCW'] = False
        danger_zone_y = h * FCW_DANGER_ZONE_Y
        center_x_min = w * FCW_CENTER_X_MIN
        center_x_max = w * FCW_CENTER_X_MAX
        
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                if cls_id in [2, 3, 5, 7]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    box_center_x = (x1 + x2) / 2
                    # Check if vehicle is close and in center lane
                    if y2 > danger_zone_y and (center_x_min < box_center_x < center_x_max):
                        self.warning_flags['FCW'] = True
        
        # Process Lane Masks
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            xy_segments = result.masks.xy
            
            left_candidates = []
            right_candidates = []
            
            # Separate masks into left/right based on centroid
            for i, segment in enumerate(xy_segments):
                if len(segment) == 0:
                    continue
                
                centroid_x = np.mean(segment[:, 0])
                centroid_y = np.mean(segment[:, 1])
                
                # Ignore upper portion (sky/horizon)
                if centroid_y < h * 0.4:
                    continue
                
                # Classify as left or right lane
                if centroid_x < mid_x:
                    left_candidates.append(segment)
                else:
                    right_candidates.append(segment)
            
            # Fit lines with weighted least squares
            if left_candidates:
                all_left = np.vstack(left_candidates)
                # Weight points closer to vehicle (bottom of image) more heavily
                weights = (all_left[:, 1] / h) ** 2  # Quadratic weighting
                result = self._fit_line_weighted(
                    np.column_stack([all_left[:, 1], all_left[:, 0]]),
                    weights=weights
                )
                if result:
                    m_l, b_l = result
                    self.left_tracker.update(m_l, b_l)
            
            if right_candidates:
                all_right = np.vstack(right_candidates)
                weights = (all_right[:, 1] / h) ** 2
                result = self._fit_line_weighted(
                    np.column_stack([all_right[:, 1], all_right[:, 0]]),
                    weights=weights
                )
                if result:
                    m_r, b_r = result
                    self.right_tracker.update(m_r, b_r)
            
            # Store debug mask if enabled
            if SHOW_DEBUG_MASKS and len(masks) > 0:
                self.debug_mask = masks[0]

    def _update_warnings(self, mL, bL, mR, bR, h, w, mid_x):
        """
        Update LDW (Lane Departure Warning) based on filtered lane parameters.
        """
        # Calculate lane positions at bottom of image (y = h)
        x_left_bottom = mL * h + bL
        x_right_bottom = mR * h + bR
        
        # Lane center in pixels
        lane_center_px = (x_left_bottom + x_right_bottom) / 2
        
        # Vehicle offset in pixels (positive = right of center)
        offset_px = lane_center_px - mid_x
        
        # Convert to meters using lane width calibration
        lane_width_px = abs(x_right_bottom - x_left_bottom)
        if lane_width_px > 50:  # Sanity check
            self.vehicle_offset_m = (offset_px / lane_width_px) * LANE_WIDTH_M
        else:
            self.vehicle_offset_m = 0.0
        
        # LDW: Trigger if offset exceeds threshold
        if abs(self.vehicle_offset_m) > LDW_OFFSET_THRESHOLD_M:
            self.warning_flags['LDW'] = True
        else:
            self.warning_flags['LDW'] = False

    def draw_visualization(self, frame, left_params, right_params):
        """
        Draw lane overlay, warnings, and HUD on frame.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        mL, bL = left_params
        mR, bR = right_params
        
        # Determine color based on warnings
        if self.warning_flags['LDW']:
            lane_color = LANE_COLOR_DANGER
        elif abs(self.vehicle_offset_m) > LDW_OFFSET_THRESHOLD_M * 0.7:
            lane_color = LANE_COLOR_WARNING
        else:
            lane_color = LANE_COLOR_GOOD
        
        # Draw lane lines
        def draw_lane_line(m, b, color, thickness=6):
            y1 = h
            y2 = int(h * 0.5)
            x1 = int(m * y1 + b)
            x2 = int(m * y2 + b)
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
        
        if self.left_tracker.is_initialized and self.left_tracker.get_confidence() > 0.3:
            draw_lane_line(mL, bL, lane_color)
        
        if self.right_tracker.is_initialized and self.right_tracker.get_confidence() > 0.3:
            draw_lane_line(mR, bR, lane_color)
        
        # Draw filled lane area
        if (self.left_tracker.is_initialized and self.right_tracker.is_initialized and
            self.left_tracker.get_confidence() > 0.3 and self.right_tracker.get_confidence() > 0.3):
            
            y_vals = np.linspace(h * 0.5, h, 30)
            l_vals = mL * y_vals + bL
            r_vals = mR * y_vals + bR
            
            pts = np.zeros((len(y_vals) * 2, 2), dtype=np.int32)
            for i in range(len(y_vals)):
                pts[i] = [int(l_vals[i]), int(y_vals[i])]
                pts[len(pts) - 1 - i] = [int(r_vals[i]), int(y_vals[i])]
            
            cv2.fillPoly(overlay, [pts], lane_color)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # HUD: System Status
        status_text = "MOVING" if self.motion_detector.is_moving else "STATIONARY (Inference Paused)"
        status_color = (100, 255, 100) if self.motion_detector.is_moving else (150, 150, 150)
        cv2.putText(frame, f"Status: {status_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # HUD: Performance
        cv2.putText(frame, f"Inference: {self.inference_fps:.1f} FPS", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # HUD: Lane Offset
        offset_color = LANE_COLOR_DANGER if abs(self.vehicle_offset_m) > LDW_OFFSET_THRESHOLD_M else (0, 255, 255)
        cv2.putText(frame, f"Lane Offset: {self.vehicle_offset_m:+.2f}m", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)
        
        # HUD: Tracker Confidence
        left_conf = self.left_tracker.get_confidence()
        right_conf = self.right_tracker.get_confidence()
        cv2.putText(frame, f"Track: L={left_conf:.2f} R={right_conf:.2f}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Warnings
        if self.warning_flags['LDW']:
            cv2.putText(frame, "⚠ LANE DEPARTURE WARNING", (w//2 - 220, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, LANE_COLOR_DANGER, 3)
            cv2.rectangle(frame, (10, 10), (w-10, h-10), LANE_COLOR_DANGER, 4)
        
        if self.warning_flags['FCW']:
            cv2.putText(frame, "⚠ FORWARD COLLISION RISK", (w//2 - 220, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, LANE_COLOR_DANGER, 3)
        
        # Debug: Show segmentation mask
        if SHOW_DEBUG_MASKS and self.debug_mask is not None:
            mask_resized = cv2.resize(self.debug_mask, (w, h))
            mask_colored = cv2.applyColorMap((mask_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            frame[0:h//4, 0:w//4] = cv2.resize(mask_colored, (w//4, h//4))
        
        return frame


# ====== Main Execution ======
def main():
    parser = argparse.ArgumentParser(
        description="Simple Pilot Pro - Optimized ADAS with Kalman Filtering and OpenVINO"
    )
    parser.add_argument("video_path", nargs='?', type=str, help="Path to video file (optional)")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--start_sec", type=float, default=0.0, help="Skip first N seconds (video only)")
    parser.add_argument("--no-openvino", action='store_true', help="Disable OpenVINO optimization")
    parser.add_argument("--debug", action='store_true', help="Show debug visualizations")
    args = parser.parse_args()
    
    # Enable debug mode if requested
    global SHOW_DEBUG_MASKS
    if args.debug:
        SHOW_DEBUG_MASKS = True
    
    # Initialize video source
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'smart_pilot_{timestamp}.mp4'
    
    if args.video_path:
        source = cv2.VideoCapture(args.video_path)
        use_camera = False
        if args.start_sec > 0:
            print(f"⏩ Skipping first {args.start_sec} seconds...")
            source.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000)
        print(f"📹 Processing video: {args.video_path}")
    else:
        try:
            source = VideoStream(args.camera, 1280, 720, 30.0, output_filename)
            source.start()
            use_camera = True
            print(f"🎥 Recording to: {output_filename}")
        except Exception as e:
            print(f"✗ Camera Error: {e}")
            return
    
    # Initialize lane detection system
    print("\n🚀 Initializing Lane Detection System...")
    use_openvino = not args.no_openvino
    pilot = LaneDetectionSystem(use_openvino=use_openvino)
    
    if pilot.model is None:
        print("✗ Failed to load model. Exiting.")
        return
    
    print("✓ System ready!\n")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'd' - Toggle debug mode\n")
    
    cv2.namedWindow("Simple Pilot Pro", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    fps_start = time.time()
    display_fps = 0
    
    try:
        while True:
            # Read frame
            if use_camera:
                frame = source.read()
            else:
                ret, frame = source.read()
                if not ret:
                    break
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Process frame
            frame, l_params, r_params = pilot.process_frame(frame)
            
            # Draw visualization
            final_frame = pilot.draw_visualization(frame, l_params, r_params)
            
            # Calculate display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end = time.time()
                display_fps = 30 / (fps_end - fps_start)
                fps_start = fps_end
            
            # Display FPS on frame
            cv2.putText(final_frame, f"Display: {display_fps:.1f} FPS", (20, final_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Show frame
            cv2.imshow("Simple Pilot Pro", final_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"screenshot_{timestamp}_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, final_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord('d'):
                SHOW_DEBUG_MASKS = not SHOW_DEBUG_MASKS
                print(f"🔧 Debug mode: {'ON' if SHOW_DEBUG_MASKS else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    finally:
        print("\n🛑 Shutting down...")
        if use_camera:
            source.stop()
        else:
            source.release()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()
