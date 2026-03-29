import cv2
import numpy as np
import time
import argparse
import os
from collections import deque
from threading import Thread, Lock
from queue import Queue
from datetime import datetime

# --- Config & Imports ---
# Workaround for OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' library not found. Please install with: pip install ultralytics")
    exit(1)

# ====== Configuration ======
# Inference Settings
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
LANE_MODEL_NAME = "yolov8n-seg.pt" # Or path to custom trained lane model
OPENVINO_DIR = "yolov8n-seg_openvino_model" # Exported OpenVINO model directory
INFERENCE_SIZE = 640 # Match your OpenVINO model export size

# Motion Detection Settings
MOTION_THRESHOLD = 2.0  # Pixel shift threshold
STATIONARY_FRAME_COUNT = 15 # Frames to wait before declaring stationary
MOTION_CHANGE_RATIO = 0.008 # 0.8% change threshold

# Warning Thresholds (Perspective View heuristics)
LDW_THRESHOLD_PX = 100 # Pixels from center
FCW_DISTANCE_THRESHOLD = 0.8 # Relative size or distance estimate

class VideoStream:
    """
    Handles camera capture and video recording in separate threads.
    Asynchronous capture to maintain high FPS recording even if inference lags.
    """
    def __init__(self, camera_idx, width, height, target_fps, output_file):
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_idx)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_idx}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Focus Lock (Infinity)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)     

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_file, fourcc, target_fps, (self.width, self.height))

        self.write_queue = Queue(maxsize=128)
        self.latest_frame = None
        self.stopped = False
        self.lock = Lock()
        
        self.capture_count = 0
        self.actual_capture_fps = 0
        
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

            if not self.write_queue.full():
                try:
                    self.write_queue.put_nowait(frame)
                except:
                    pass
            
            with self.lock:
                self.latest_frame = frame

    def _writer_loop(self):
        while not self.stopped or not self.write_queue.empty():
            if not self.write_queue.empty():
                frame = self.write_queue.get()
                # Assuming camera is mounted upside down or needs flipping as per original code
                flipped = cv2.flip(frame, -1) if frame is not None else None
                if flipped is not None:
                    self.out.write(flipped)
                self.write_queue.task_done()
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                # Flip for processing to match recorded orientation
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
    Lightweight motion detection using frame differencing / optical flow.
    Used to sleep heavy inference when vehicle is stationary.
    """
    def __init__(self, threshold=MOTION_THRESHOLD):
        self.prev_gray = None
        self.threshold = threshold
        self.stationary_counter = 0
        self.is_moving = True

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180)) # Downscale for speed
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return True

        # Calculate Frame Difference via simple Diff
        diff = cv2.absdiff(self.prev_gray, gray)
        # diff > 25 is a heuristic for meaningful change
        non_zero_count = np.count_nonzero(diff > 25) 
        
        # Heuristic: If significant pixels changed, we are moving
        # Normalize by area
        change_ratio = non_zero_count / (gray.shape[0] * gray.shape[1])
        
        if change_ratio > MOTION_CHANGE_RATIO:
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
    Kalman Filter for tracking lane line parameters (slope m, intercept b).
    State Vector: [m, b, dm, db] (Slope, Intercept, DeltaSlope, DeltaIntercept)
    Model: Constant Velocity (we assume slope/intercept change smoothly).
    """
    def __init__(self):
        # 4 State variables, 2 Measurement variables (m, b)
        self.kf = cv2.KalmanFilter(4, 2, 0)
        
        # Transition Matrix (F)
        # x_k = x_{k-1} + dx * dt
        # m = m + dm, b = b + db
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement Matrix (H)
        # We measure m and b directy
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process Noise Covariance (Q)
        # Trust the model quite a bit, but allow some drift for curvature changes
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.processNoiseCov[2, 2] = 1e-3 # Allow more variance in derivative (curve change)
        self.kf.processNoiseCov[3, 3] = 1e-3

        # Measurement Noise Covariance (R)
        # How much we trust the noisy detection vs the filter
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # Error Covariance (P)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.age = 0
        self.is_initialized = False

    def predict(self):
        pred = self.kf.predict()
        self.age += 1
        return pred

    def update(self, m, b):
        measurement = np.array([[np.float32(m)], [np.float32(b)]])
        self.kf.correct(measurement)
        self.age = 0
        self.is_initialized = True

    def get_state(self):
        return self.kf.statePost[0][0], self.kf.statePost[1][0]

    def get_confidence(self):
        if not self.is_initialized:
            return 0.0
        # Decay confidence if no measurements received (age increases)
        return max(0.0, 1.0 - (self.age * 0.05))

class LaneSystem:
    def __init__(self):
        # Load Model
        self.model = self._load_model()
        
        # Trackers for Left and Right lanes
        # Using x = my + b formulation (better for verticalish lines)
        self.left_tracker = KalmanLaneTracker()
        self.right_tracker = KalmanLaneTracker()
        
        self.motion_detector = MotionDetector()
        
        # State
        self.vehicle_offset = 0
        self.warning_flags = {'LDW': False, 'FCW': False}
        self.detected_boxes = []  # Store detected vehicle boxes
        self.inference_fps = 0
        self.last_inference_time = time.time()

    def _load_model(self):
        # Hardware Optimization: OpenVINO
        # If exported model exists, load it.
        path = OPENVINO_DIR if os.path.exists(OPENVINO_DIR) else LANE_MODEL_NAME
        print(f"Loading Model: {path}")
        try:
            model = YOLO(path, task='segment')
            # Check for Intel GPU if available (User Request)
            # Ultralytics automatic, but can force in predict arguments
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    def fit_line_from_mask(self, mask_pts):
        """
        Fit x = my + b
        Returns (m, b)
        """
        if len(mask_pts) < 50: return None
        
        y = mask_pts[:, 0] # Rows
        x = mask_pts[:, 1] # Cols
        
        # Fit polynomial x = Ay + B (Order 1)
        try:
            z = np.polyfit(y, x, 1)
            return z # [m, b]
        except:
            return None

    def process(self, frame):
        h, w = frame.shape[:2]
        mid_x = w // 2
        
        # 1. Motion Check
        moving = self.motion_detector.update(frame)
        
        # Clear previous detections
        self.detected_boxes = []
        
        if moving:
            # Predict Kalman State only when moving
            self.left_tracker.predict()
            self.right_tracker.predict()
            
            if self.model:
                inference_start = time.time()
                
                # 2. Run Inference
                # Explicit imgsz should match the exported OpenVINO model size
                use_half = True if os.path.exists(OPENVINO_DIR) else False
                results = self.model(
                    frame, 
                    verbose=False, 
                    half=use_half, 
                    device='cpu', 
                    imgsz=INFERENCE_SIZE,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD
                ) 
                
                # Calculate inference FPS
                inference_time = time.time() - inference_start
                self.inference_fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Get current lane boundaries for filtering
                mL, bL = self.left_tracker.get_state()
                mR, bR = self.right_tracker.get_state()
                
                # --- Process Vehicle Detections ---
                danger_zone_y = h * 0.65
                self.warning_flags['FCW'] = False
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # COCO IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
                        if cls_id in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            box_center_x = (x1 + x2) / 2
                            box_bottom_y = y2
                            
                            # Store box for visualization
                            self.detected_boxes.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'cls_id': cls_id,
                                'conf': conf,
                                'in_lane': False,
                                'danger': False
                            })
                            
                            # Check if vehicle is in our lane (between left and right lane lines)
                            if self.left_tracker.is_initialized and self.right_tracker.is_initialized:
                                # Calculate lane boundaries at vehicle's y position
                                x_left_at_vehicle = mL * box_bottom_y + bL
                                x_right_at_vehicle = mR * box_bottom_y + bR
                                
                                # Check if vehicle center is within lane boundaries
                                if x_left_at_vehicle < box_center_x < x_right_at_vehicle:
                                    self.detected_boxes[-1]['in_lane'] = True
                                    
                                    # FCW: Only trigger for vehicles in our lane
                                    if box_bottom_y > danger_zone_y:
                                        self.warning_flags['FCW'] = True
                                        self.detected_boxes[-1]['danger'] = True

                # 3. Parse Masks for Lane Detection
                if results[0].masks is not None:
                    # xy_segments are in original image scale
                    xy_segments = results[0].masks.xy
                    
                    left_candidates = []
                    right_candidates = []
                    
                    for i, segment in enumerate(xy_segments):
                        if len(segment) < 20: continue
                        
                        centroid_x = np.mean(segment[:, 0])
                        centroid_y = np.mean(segment[:, 1])
                        
                        # Filter: must be in lower portion of image
                        if centroid_y < h * 0.4: continue
                        
                        # Aspect Ratio Filter: Lanes are vertically elongated
                        x_min, y_min = np.min(segment, axis=0)
                        x_max, y_max = np.max(segment, axis=0)
                        if (y_max - y_min) / (x_max - x_min + 1) < 1.0: continue
                        
                        # Improved classification: Position + Relative to previous state
                        if self.left_tracker.is_initialized and self.right_tracker.is_initialized:
                            # Use distance to predicted line
                            dist_left = abs(centroid_x - (mL * centroid_y + bL))
                            dist_right = abs(centroid_x - (mR * centroid_y + bR))
                            if dist_left < dist_right:
                                left_candidates.append(segment)
                            else:
                                right_candidates.append(segment)
                        else:
                            # Fallback to mid_x
                            if centroid_x < mid_x:
                                left_candidates.append(segment)
                            else:
                                right_candidates.append(segment)
                    
                    # Fit and validate
                    valid_detections = []
                    new_mL, new_bL = None, None
                    new_mR, new_bR = None, None

                    if left_candidates:
                        all_left = np.vstack(left_candidates)
                        weights = (all_left[:, 1] / h) ** 2
                        new_mL, new_bL = np.polyfit(all_left[:, 1], all_left[:, 0], 1, w=weights)
                        
                    if right_candidates:
                        all_right = np.vstack(right_candidates)
                        weights = (all_right[:, 1] / h) ** 2
                        new_mR, new_bR = np.polyfit(all_right[:, 1], all_right[:, 0], 1, w=weights)
                    
                    # Lane Width Consistency Check (Stability improvement)
                    if new_mL is not None and new_mR is not None:
                        # Check width at bottom of image
                        width_bottom = (new_mR * h + new_bR) - (new_mL * h + new_bL)
                        # Standard road width in relative terms should be reasonable (e.g. 0.3 to 0.8 of w)
                        if 0.25 * w < width_bottom < 0.85 * w:
                            self.left_tracker.update(new_mL, new_bL)
                            self.right_tracker.update(new_mR, new_bR)
                    elif new_mL is not None:
                        self.left_tracker.update(new_mL, new_bL)
                    elif new_mR is not None:
                        self.right_tracker.update(new_mR, new_bR)
        else:
            # Stationary: Skip inference AND skip prediction to freeze lanes
            pass

        # 4. Get Smoothed Lane Estimates
        mL, bL = self.left_tracker.get_state()
        mR, bR = self.right_tracker.get_state()
        
        # 5. Lane Departure Warning (LDW)
        if self.left_tracker.is_initialized and self.right_tracker.is_initialized:
            x_left_bottom = mL * h + bL
            x_right_bottom = mR * h + bR
            lane_center = (x_left_bottom + x_right_bottom) / 2
            self.vehicle_offset = lane_center - mid_x
            
            if abs(self.vehicle_offset) > LDW_THRESHOLD_PX:
                self.warning_flags['LDW'] = True
            else:
                self.warning_flags['LDW'] = False
        else:
            self.warning_flags['LDW'] = False
            
        return frame, (mL, bL), (mR, bR)

    def draw(self, frame, left_params, right_params):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        mL, bL = left_params
        mR, bR = right_params
        
        # Draw Lane Lines
        def draw_line(m, b, color, thickness=6):
            y1 = h
            y2 = int(h * 0.5)
            x1 = int(m * y1 + b)
            x2 = int(m * y2 + b)
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
        
        lane_color = (0, 255, 0) if not self.warning_flags['LDW'] else (0, 0, 255)
        
        if self.left_tracker.is_initialized:
            draw_line(mL, bL, lane_color)
        if self.right_tracker.is_initialized:
            draw_line(mR, bR, lane_color)
            
        # Draw Lane Area (only if consistent)
        if self.left_tracker.is_initialized and self.right_tracker.is_initialized:
            if self.left_tracker.get_confidence() > 0.4 and self.right_tracker.get_confidence() > 0.4:
                y_vals = np.linspace(h*0.5, h, 20)
                l_vals = mL * y_vals + bL
                r_vals = mR * y_vals + bR
                
                pts = np.zeros((len(y_vals) * 2, 2), dtype=np.int32)
                for i in range(len(y_vals)):
                    pts[i] = [int(l_vals[i]), int(y_vals[i])]
                    pts[len(pts) - 1 - i] = [int(r_vals[i]), int(y_vals[i])]
                
                cv2.fillPoly(overlay, [pts], lane_color)
        
        # Blend lane overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw Vehicle Detection Boxes
        for detection in self.detected_boxes:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['conf']
            in_lane = detection['in_lane']
            danger = detection['danger']
            
            # Color coding: Red for danger, Yellow for in-lane, Green for out-of-lane
            if danger:
                color = (0, 0, 255)  # Red - Collision risk
                thickness = 3
            elif in_lane:
                color = (0, 255, 255)  # Yellow - In our lane
                thickness = 2
            else:
                color = (0, 255, 0)  # Green - Adjacent lane
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence label
            label = f"{conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # HUD - System Status
        status = "MOVING" if self.motion_detector.is_moving else "STATIONARY (Paused)"
        status_color = (0, 255, 0) if self.motion_detector.is_moving else (150, 150, 150)
        cv2.putText(frame, f"Status: {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # HUD - FPS
        cv2.putText(frame, f"Inference: {self.inference_fps:.1f} FPS", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # HUD - Lane Offset
        off_m = (self.vehicle_offset / w) * 3.7
        offset_color = (0, 255, 255) if abs(off_m) < 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Offset: {off_m:+.2f}m", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)
        
        # HUD - Detection Count
        in_lane_count = sum(1 for d in self.detected_boxes if d['in_lane'])
        cv2.putText(frame, f"Vehicles: {len(self.detected_boxes)} ({in_lane_count} in lane)", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Warnings
        if self.warning_flags['LDW']:
            cv2.putText(frame, "⚠ LANE DEPARTURE", (w//2 - 150, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        if self.warning_flags['FCW']:
            cv2.putText(frame, "⚠ COLLISION RISK", (w//2 - 150, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            # Draw danger zone indicator
            danger_y = int(h * 0.65)
            cv2.line(frame, (0, danger_y), (w, danger_y), (0, 0, 255), 2)

        return frame

# ====== Main Execution ======
def main():
    parser = argparse.ArgumentParser(description="Smart Pilot - Perspective & Kalman")
    parser.add_argument("video_path", nargs='?', type=str, help="Path to video file")
    parser.add_argument("--camera", type=int, default=1, help="Camera Index")
    parser.add_argument("--start_sec", type=float, default=0.0, help="Skip first N seconds (video only)")
    args = parser.parse_args()

    # Initialize Video Source
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'smart_pilot_{timestamp}.mp4'
    
    if args.video_path:
        source = cv2.VideoCapture(args.video_path)
        use_camera = False
        if args.start_sec > 0:
            print(f"Skipping first {args.start_sec} seconds...")
            source.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000)
    else:
        try:
            source = VideoStream(args.camera, 1280, 720, 30.0, output_filename)
            source.start()
            use_camera = True
            print(f"Recording to {output_filename}")
        except Exception as e:
            print(f"Camera Error: {e}")
            return

    # Initialize Logic
    pilot = LaneSystem()
    
    cv2.namedWindow("Smart Pilot AI")
    
    try:
        while True:
            if use_camera:
                frame = source.read()
            else:
                ret, frame = source.read()
                if not ret: break
            
            if frame is None:
                time.sleep(0.01)
                continue
                
            # Process Frame
            frame, l_params, r_params = pilot.process(frame)
            
            # Draw
            final_frame = pilot.draw(frame, l_params, r_params)
            
            cv2.imshow("Smart Pilot AI", final_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        if source and use_camera:
            source.stop()
        elif source:
            source.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
