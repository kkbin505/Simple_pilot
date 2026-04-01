import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import time
import argparse
import json  # Added for loading Gyroflow calibration
from collections import deque

# Fix for OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Global configuration
# LPF_ALPHA is no longer used, we now use LaneTracker(history_len=10)

# ====== Front Vehicle Detection ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("yolov8n.pt")  # Official Ultralytics YOLOv8

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# ====== Lens Calibration Loader (Gyroflow JSON) ======
class CameraCalibrator:
    def __init__(self, json_path):
        if not os.path.exists(json_path):
            print(f"⚠️ Calibration file not found: {json_path}")
            self.map1, self.map2 = None, None
            return

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Gyroflow stores fisheye params under "fisheye_params"
        if "fisheye_params" not in data:
            print("⚠️ Invalid Gyroflow JSON format (missing fisheye_params)")
            self.map1, self.map2 = None, None
            return

        cam_data = data["fisheye_params"]
        self.k = np.array(cam_data["camera_matrix"], dtype=np.float32)
        self.d = np.array(cam_data["distortion_coeffs"], dtype=np.float32).reshape(1, 4)
        self.cal_dim = (data["calib_dimension"]["w"], data["calib_dimension"]["h"])
        
        
        # Pre-compute maps once for efficiency
        # We use fisheye model as indicated by Gyroflow JSON
        # balance=0.5 means a good trade-off between keeping FOV and avoiding excessive stretching
        self.new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.k, self.d, self.cal_dim, np.eye(3), balance=0.5
        )

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.k, self.d, np.eye(3), self.new_k, self.cal_dim, cv2.CV_16SC2
        )
        print(f"✅ Loaded lens calibration: {data['name']}")

    def undistort(self, img):
        if self.map1 is not None and self.map2 is not None:
            # Resize if input frame doesn't match calibration dimensions
            h, w = img.shape[:2]
            if (w, h) != self.cal_dim:
                img = cv2.resize(img, self.cal_dim)
            # Use INTER_CUBIC for sharper edges than INTER_LINEAR
            return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_CUBIC)
        return img

    def undistort_points(self, points):
        """Transform distorted pixel points to undistorted pixel space."""
        if self.map1 is None or len(points) == 0:
            return points
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        # Convert to normalized coordinates and then to new camera matrix space
        undistorted = cv2.fisheye.undistortPoints(points, self.k, self.d, R=np.eye(3), P=self.new_k)
        return undistorted.reshape(-1, 2)

    def distort_points(self, points):
        """Transform undistorted pixel points back to distorted pixel space for overlay."""
        if self.map1 is None or len(points) == 0:
            return points
        points = np.array(points, dtype=np.float32)
        # Convert back to normalized coordinates by inverting new_k manually
        # new_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = self.new_k[0, 0]
        fy = self.new_k[1, 1]
        cx = self.new_k[0, 2]
        cy = self.new_k[1, 2]
        
        pts_norm = np.zeros((len(points), 1, 3), dtype=np.float32)
        pts_norm[:, 0, 0] = (points[:, 0] - cx) / fx
        pts_norm[:, 0, 1] = (points[:, 1] - cy) / fy
        pts_norm[:, 0, 2] = 1.0  # Set Z=1 for projection
        
        # Project back using the original distortion model
        projected, _ = cv2.fisheye.projectPoints(pts_norm, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), self.k, self.d)
        return projected.reshape(-1, 2)

    def get_scaled_k(self, current_width):
        """
        Scale the intrinsic matrix new_k to match the current frame resolution.
        """
        scale = current_width / self.cal_dim[0]
        scaled_k = self.new_k.copy()
        scaled_k[0, 0] *= scale # fx
        scaled_k[1, 1] *= scale # fy
        scaled_k[0, 2] *= scale # cx
        scaled_k[1, 2] *= scale # cy
        return scaled_k

class LaneTracker:
    """Historical window-based lane smoothing tracker"""
    
    def __init__(self, history_len=10):
        # Store poly coefficients (ax^2 + bx + c or mx + c)
        self.left_history = deque(maxlen=history_len)
        self.right_history = deque(maxlen=history_len)
        self.last_valid = {'left': None, 'right': None}
    
    def update(self, left_poly, right_poly):
        """
        Update with new detection and return smoothed coefficients
        """
        if left_poly is not None:
            self.left_history.append(left_poly)
            self.last_valid['left'] = left_poly
            
        if right_poly is not None:
            self.right_history.append(right_poly)
            self.last_valid['right'] = right_poly
            
        smoothed = {'left': None, 'right': None}
        
        # Calculate moving average across the window
        for side, history in [('left', self.left_history), ('right', self.right_history)]:
            if len(history) > 0:
                # Average all poly coefficients in the queue
                # Use np.mean along axis 0 since history items are arrays
                smoothed[side] = np.mean(list(history), axis=0)
            else:
                # Return last known valid result if no history exists
                smoothed[side] = self.last_valid[side]
                
        return smoothed['left'], smoothed['right']

# ====== Lane Detection and LDW Functions ======
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    width_ratio = 0.6
    width_extention = -100
    left_offset = 0
    roi_top = 0.65 # Keep only bottom 30% to exclude sky
    roi_bottom = 0.9 # Exclude bottom 10% (hood)
    y_bottom = int(height * roi_bottom)
    y_top = int(height * roi_top)
    
    polygon = np.array([[(-width_extention+left_offset, y_bottom), (width+width_extention, y_bottom), 
                         (int(width*width_ratio), y_top), (int(width*(1-width_ratio))+left_offset, y_top)]])
    cv2.fillPoly(mask,polygon,(255, 255, 255))
    return cv2.bitwise_and(img, mask)

def draw_ldw_warning(img, deviation, threshold=50):
    """
    Draw a directional warning overlay at the lower 1/3 of the screen.
    """
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # Target Y coordinate: roughly 1/3 above the bottom (2/3 down from top)
    # This places the HUD right above the vehicle's hood/dashboard area
    hud_y = int(h * 0.75)
    
    # Check if deviation exceeds the limit
    if abs(deviation) > threshold:
        color = (0, 0, 255) # Red for warning
        
        if deviation > 0:
            msg = "! STEER LEFT <--"
            # Draw a horizontal warning bar at the HUD level
            cv2.rectangle(overlay, (w // 2, hud_y - 45), (w - 50, hud_y + 15), color, -1)
        else:
            msg = "--> STEER RIGHT !"
            # Draw a horizontal warning bar at the HUD level
            cv2.rectangle(overlay, (50, hud_y - 45), (w // 2, hud_y + 15), color, -1)
            
        # Blend the semi-transparent warning bars
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Draw warning text with a small shadow for readability
        text_x = w // 2 - 160
        # Shadow
        cv2.putText(img, msg, (text_x + 2, hud_y + 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)
        # Main Text
        cv2.putText(img, msg, (text_x, hud_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    else:
        # Normal state - Show offset in green
        offset_text = f"Off: {deviation:+}px"
        cv2.putText(img, offset_text, (w // 2 - 60, hud_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
    return img

def draw_lines(img, lines, calibrator=None, tracker=None):
    line_img = np.zeros_like(img)
    if lines is None:
        return img, None, None

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        # Set to 0.5 to filter out road textures and horizontal artifacts
        if abs(slope) < 0.5:
            continue
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

    # --- Fitting Logic ---
    left_poly_raw = None
    right_poly_raw = None

    for side, lines_list in [('left', left_lines), ('right', right_lines)]:
        if len(lines_list) == 0:
            continue
            
        x, y = [], []
        for x1, y1, x2, y2 in lines_list:
            x += [x1, x2]
            y += [y1, y2]
        
        pts = np.column_stack((x, y))
        
        # --- Wide Angle Correction on Extracted Points ---
        # if calibrator:
        #     pts = calibrator.undistort_points(pts)
        
        poly = np.polyfit(pts[:, 1], pts[:, 0], 1)
        if side == 'left':
            left_poly_raw = poly
        else:
            right_poly_raw = poly

    # --- History Smoothing using Tracker ---
    if tracker:
        left_poly, right_poly = tracker.update(left_poly_raw, right_poly_raw)
    else:
        left_poly, right_poly = left_poly_raw, right_poly_raw

    # --- Draw Results ---
    height = img.shape[0]
    y_range = np.linspace(int(height * 0.65), height, 20)
    lane_info = {'left': None, 'right': None}

    for side, poly in [('left', left_poly), ('right', right_poly)]:
        if poly is not None:
            x_pts = np.polyval(poly, y_range)
            pts = np.column_stack((x_pts, y_range))
            
            # # If we used calibration to fit, we must project back to distort space for drawing
            # if calibrator:
            #     draw_pts = calibrator.distort_points(pts)
            #     # For LDW, save the undistorted bottom point (at y = img_height)
            #     lane_info[side] = (int(np.polyval(poly, height)), height)
            # else:
            draw_pts = pts
            lane_info[side] = (int(draw_pts[-1][0]), height)

            # Draw the curve/line
            draw_pts = draw_pts.astype(np.int32)
            for i in range(len(draw_pts) - 1):
                cv2.line(line_img, tuple(draw_pts[i]), tuple(draw_pts[i+1]), (255, 0, 0), 6)

    # Blend lines with original frame
    line_mask = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) > 0
    output = img.copy()
    output[line_mask] = cv2.addWeighted(img, 0.4, line_img, 0.6, 0)[line_mask]
    
    return output, lane_info['left'], lane_info['right']

# ====== Main Loop ======
def main():
    parser = argparse.ArgumentParser(description="Simple Pilot: Vision-Based Driving Assistant")
    parser.add_argument("video_source", nargs='?', default="0", help="Path to video file or camera index (default: 0 for webcam)")
    parser.add_argument("-s", "--start", type=float, default=0, help="Start time in seconds for video analysis")
    parser.add_argument("--lens", type=str, default=r"lens\Logitech_C920__Auto_1080p_16by9_1920x1080-30.00fps.json", help="Path to Gyroflow calibration JSON")
    args = parser.parse_args()

    # Determine if source is video file or camera index
    source = args.video_source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    # Initialize Components
    calibrator = CameraCalibrator(args.lens) if args.lens else None
    tracker = LaneTracker(history_len=10)

    # Try setting capture resolution to match calibration if it's a camera
    if isinstance(source, int) and calibrator:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, calibrator.cal_dim[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calibrator.cal_dim[1])

    # Seek to start time if specified
    if args.start > 0:
        # print(f"⏭️ Skipping to {args.start} seconds...")
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    prev_time = 0  # <- Initialize FPS timer
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Lens Correction
        # if calibrator:
        #     frame = calibrator.undistort(frame)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        h, w = frame.shape[:2]

        # ----- Lane Detection + LDW -----
        # 1. CLAHE for localized contrast enhancement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 2. HSV Filtering for grey-white and yellow lanes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Grey-white: Low Saturation (S), High brightness (V)
        lower_grey_white = np.array([0, 0, 150])
        upper_grey_white = np.array([180, 55, 255])
        white_mask = cv2.inRange(hsv, lower_grey_white, upper_grey_white)
        
        # Yellow: Specific Hue (H) range
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # cv2.imshow("Color Mask", color_mask)

        # 3. Combine Canny Edges with Color Mask
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        canny_edges = cv2.Canny(blur, 80, 200)

        # 1. Combine FULL edges and color mask first (Fast)
        edges = cv2.bitwise_or(canny_edges, color_mask)
        
        # 2. Apply ROI mask ONCE to the final combined binary image
        roi = region_of_interest(edges)
        
        # Verify: this should ONLY show the road lanes, no sky artifacts!
        # cv2.imshow("Clean ROI", roi)
        # roi = region_of_interest(edges)

        # 2. Rectify AFTER masking (Ensures straight lines for Hough and LDW)
        # if calibrator:
        #     roi = calibrator.undistort(roi)
        #     frame = calibrator.undistort(frame)

        hough_lines = cv2.HoughLinesP(roi, 1, np.pi/180, 60, minLineLength=100, maxLineGap=100)

        
        # Debug: Draw ALL raw segments to see what Hough detected
        debug_img = np.zeros_like(frame)

        # print(hough_lines)
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1), (x2, y2), 255, 2)
        # cv2.imshow("Hough Raw Segments", debug_img)

        # undistorted_frame = calibrator.undistort(frame)
        
        # Lane Detection and tracking
        output, left_lane, right_lane = draw_lines(frame, hough_lines, calibrator, tracker)
        
        # LDW
        # Directional Lane Departure Warning (LDW)
        if left_lane and right_lane:
            # Calculate frame center with automatic scale adjustment
            if calibrator:
                # Scale the principal point from 1920 (or calib_dim) to the current frame width
                scaled_k = calibrator.get_scaled_k(w)
                frame_center_base = scaled_k[0, 2]
            else:
                frame_center_base = w // 2
            
            # Manual center fix (should be 0 now after auto-scaling)
            CENTER_FIX = -60
            frame_center = frame_center_base + CENTER_FIX
            
            # Calculate lane center at the bottom of the visible area
            mid_bottom_x = (left_lane[0] + right_lane[0]) // 2
            
            # Deviation: Positive means the car is to the right of the lane center
            deviation = int(frame_center - mid_bottom_x)
            
            # Apply the directional warning overlay
            # Using a threshold of 50px as discussed
            output = draw_ldw_warning(output, deviation, threshold=50)
            
            # Optional: Draw the center alignment lines for debug
            cv2.line(output, (int(frame_center), h), (int(frame_center), h - 100), (255, 255, 0), 2)
            cv2.line(output, (mid_bottom_x, h), (mid_bottom_x, h - 80), (0, 255, 0), 2)

        # ----- YOLO Front Vehicle Detection + FCW -----
        results = yolo_model(frame, conf=0.4)[0]

        warning_fcw = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height_ratio = (y2 - y1) / h

            if box_height_ratio > 0.35:  # Simple distance approximation
                color = (0,0,255)
                warning_fcw = True
            else:
                color = (0,255,0)

            cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if warning_fcw:
            cv2.putText(output, "FORWARD COLLISION WARNING!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time  # <- Update last frame time

        cv2.putText(output, f"FPS: {fps:.1f}", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        new_width = 800
        new_height = 450
        resized_img = cv2.resize(output, (new_width, new_height))
        cv2.imshow("Simple Pilot - LDW + FCW", resized_img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
