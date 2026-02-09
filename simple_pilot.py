
import cv2
import numpy as np
import torch
import os
import time
from collections import deque

# Workaround for OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO

# ====== Configuration ======
VIDEO_PATH = "Laguna_road_20260208.mp4"
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# ====== Model Loading ======
# Using OpenVINO model for Intel acceleration
yolo_model = YOLO("yolov8n_openvino_model", task='detect') 

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
    cap = cv2.VideoCapture(VIDEO_PATH)
    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
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
            if yolo_model.names[cls_id] in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(output, yolo_model.names[cls_id], (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 4. Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(output, f"FPS: {fps:.1f}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Simple Pilot - Smooth Tracking", output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
