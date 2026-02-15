import cv2
import numpy as np
import torch
import os
import time
import argparse
from collections import deque
from threading import Thread, Lock
from queue import Queue
from datetime import datetime

# Workaround for OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# ====== Configuration ======
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# ====== Model Loading ======
# Using OpenVINO model for Intel acceleration
# Ensure the model path is correct relative to execution
yolo_model = YOLO("yolov8n_openvino_model", task='detect') 

class VideoStream:
    """
    Handles camera capture and video recording in separate threads.
    - Capture Thread: Pulls raw frames from camera as fast as possible.
    - Writer Thread: Processes (flips) and writes frames to disk from a queue.
    This ensures that even if writing to disk is slow, we don't drop camera frames.
    """
    def __init__(self, camera_idx, width, height, target_fps, output_file):
        # Camera Setup
        self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_idx)
        
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_idx}")

        # Try to force MJPG (Crucial for 30fps at 720p on C920)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Attempt to set FPS to 30
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # --- C920 Focus Locking ---
        # 0 is typically infinity on Windows/DSHOW for C920
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)     

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps_config = self.cap.get(cv2.CAP_PROP_FPS)

        # Video Writer Setup
        dll_found = any('openh264' in f for f in os.listdir('.'))
        if dll_found:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.codec = "avc1"
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.codec = "mp4v"

        self.out = cv2.VideoWriter(output_file, fourcc, target_fps, (self.width, self.height))
        if not self.out.isOpened():
            raise Exception("Could not open VideoWriter")

        # Threading state
        self.write_queue = Queue(maxsize=128) # Buffer up to 128 frames
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

            # Put frame in queue for writer (don't block capture if queue is full, just drop old ones)
            if not self.write_queue.full():
                self.write_queue.put(frame)
            
            # Update latest for display
            with self.lock:
                self.latest_frame = frame

    def _writer_loop(self):
        while not self.stopped or not self.write_queue.empty():
            if not self.write_queue.empty():
                frame = self.write_queue.get()
                
                # Flip 180 (Vertical + Horizontal)
                flipped = cv2.flip(frame, -1)
                
                # Resize if needed
                if flipped.shape[1] != self.width or flipped.shape[0] != self.height:
                    flipped = cv2.resize(flipped, (self.width, self.height))

                # Write to disk
                self.out.write(flipped)
                self.write_queue.task_done()
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            # Return flipped frame for display/inference to match recording
            if self.latest_frame is not None:
                return cv2.flip(self.latest_frame, -1)
            return None

    def stop(self):
        self.stopped = True
        self.capture_thread.join(timeout=1)
        self.writer_thread.join(timeout=2)
        self.cap.release()
        self.out.release()

class LaneTracker:
    """
    Tracks lane lines over time to smooth out detection jitter.
    Uses a history buffer (deque) to average line coefficients.
    """
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.left_fit_history = deque(maxlen=buffer_size)
        self.right_fit_history = deque(maxlen=buffer_size)

    def update(self, left_fit, right_fit):
        """Update history with new detection results."""
        if left_fit is not None:
            self.left_fit_history.append(left_fit)
        
        if right_fit is not None:
            self.right_fit_history.append(right_fit)

    def get_averaged_lines(self):
        """Return the smoothed (averaged) line coefficients."""
        left_avg = np.mean(self.left_fit_history, axis=0) if self.left_fit_history else None
        right_avg = np.mean(self.right_fit_history, axis=0) if self.right_fit_history else None
        return left_avg, right_avg

# Global tracker instance
tracker = LaneTracker(buffer_size=5)

def region_of_interest(img):
    """
    Applies a trapezoidal mask to focus on the road area.
    """
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    
    # Define polygon for ROI (optimized for dashboard camera view)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.58)),
        (int(width * 0.45), int(height * 0.58)),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def draw_poly_lines(img, left_fit, right_fit):
    """
    Draws smooth polynomial lines (curves) based on fit coefficients.
    """
    line_img = np.zeros_like(img)
    height = img.shape[0]
    ploty = np.linspace(int(height * 0.6), height - 1, height)

    if left_fit is not None:
        left_fitx = left_fit[0] * ploty + left_fit[1]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], np.int32)
        cv2.polylines(line_img, pts_left, False, (0, 255, 0), 8)

    if right_fit is not None:
        right_fitx = right_fit[0] * ploty + right_fit[1]
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))], np.int32)
        cv2.polylines(line_img, pts_right, False, (0, 255, 0), 8)

    return cv2.addWeighted(img, 1.0, line_img, 0.6, 0)

def detect_lines_basic(frame):
    """
    Basic lane detection pipeline: Color -> Edge -> Hough -> Fit
    """
    # 1. HLS Color Filtering (Enhances yellow/white visibility)
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    # White mask (lowered threshold for aging lines)
    white_mask = cv2.inRange(hls, np.array([0, 100, 0]), np.array([180, 255, 255]))
    # Yellow mask
    yellow_mask = cv2.inRange(hls, np.array([10, 30, 80]), np.array([45, 204, 255]))
    
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 2. Canny Edge Detection (Structure)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 3. Combine Color & Edges
    # Strategy: OR logic to catch either strong edges OR strong colors
    lane_edges = cv2.bitwise_or(edges, color_mask)
    
    # 4. ROI Masking
    roi = region_of_interest(lane_edges)
    
    # 5. Hough Transform
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=150)
    
    left_fit = None
    right_fit = None
    
    if lines is not None:
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter noise based on slope
            if abs(slope) < 0.4 or abs(slope) > 2.0: continue
            
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))

        # Linear Fit (Degree 1 Polynomial) for Left/Right groups
        def get_poly_fit(lines_list):
            if not lines_list: return None
            x, y = [], []
            for x1, y1, x2, y2 in lines_list:
                # IMPORTANT: We swap x and y for polyfit (x = f(y))
                # This is because lane lines are vertical-ish
                x += [x1, x2]
                y += [y1, y2]
            try:
                return np.polyfit(y, x, 1) # Fit x as function of y
            except:
                return None

        left_fit = get_poly_fit(left_lines)
        right_fit = get_poly_fit(right_lines)

    return left_fit, right_fit

def main():
    parser = argparse.ArgumentParser(description="Simple Pilot with C920 Support")
    parser.add_argument("video_path", nargs='?', type=str, help="Path to video file (optional)")
    parser.add_argument("--camera", type=int, default=1, help="Camera Index (default 1 for external USB)")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording FPS")
    args = parser.parse_args()

    # Generate Output Filename (only used for camera recording)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'output_{timestamp}.mp4'

    source = None
    use_camera = False

    if args.video_path:
        print(f"Processing video file: {args.video_path}")
        source = cv2.VideoCapture(args.video_path)
        use_camera = False
        if not source.isOpened():
            print(f"Error: Could not open video file {args.video_path}")
            return
    else:
        print(f"Initializing Camera {args.camera}...")
        try:
            # Initialize VideoStream with C920 optimization & Focus Lock
            source = VideoStream(args.camera, 1280, 720, args.fps, output_filename)
            source.start()
            use_camera = True
            print(f"Camera started. Recording to {output_filename}")
            print("Focus locked to infinity (0).")
        except Exception as e:
            print(f"Error starting stream: {e}")
            return

    prev_time = time.time()
    
    try:
        while True:
            if use_camera:
                # Read frame (already flipped 180 by stream.read())
                frame = source.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
            else:
                # Read from file
                ret, frame = source.read()
                if not ret: break

            # 1. Vehicle Detection (OpenVINO Optimized)
            # Using 640px for balance between speed and accuracy
            yolo_results = yolo_model(frame, conf=0.4, imgsz=640, verbose=False)[0]
            
            # 2. Lane Detection with Smoothing
            try:
                current_left, current_right = detect_lines_basic(frame)
                tracker.update(current_left, current_right)
                
                # Get smoothed lines from history
                smooth_left, smooth_right = tracker.get_averaged_lines()
                output = draw_poly_lines(frame, smooth_left, smooth_right)
                
            except Exception as e:
                print(f"Lane detection error: {e}")
                output = frame.copy()

            # 3. Draw Vehicle Bounding Boxes
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                if cls_id < len(yolo_model.names) and yolo_model.names[cls_id] in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(output, yolo_model.names[cls_id], (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 4. Display FPS and Status
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            cv2.putText(output, f"Inference FPS: {fps:.1f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if use_camera:
                cv2.putText(output, f"Capture FPS: {source.actual_capture_fps:.1f}", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(output, "REC", (1150, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                 cv2.putText(output, "FILE", (1150, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Simple Pilot - C920 Integrated", output)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(screenshot_filename, output)
                print(f"Screenshot saved: {screenshot_filename}")
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Stopping stream...")
        if use_camera:
            source.stop()
        else:
            source.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
