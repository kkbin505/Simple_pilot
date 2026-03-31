import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import time
import argparse
import json  # Added for loading Gyroflow calibration

# Fix for OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# prev_time = 0  <- moved inside main or managed globally

LPF_ALPHA = 0.15  # Smoothing factor: 1.0 = no filter, 0.05 = hyper smooth/slow
lane_poly_history = {'left': None, 'right': None}

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
        # balance=1.0 means KEEP ALL pixels (no cropping), which results in black borders
        # To avoid compression, we increase resolution by 50%
        # self.out_dim = (int(self.cal_dim[0]*1.5), int(self.cal_dim[1]*1.5))
        
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.k, self.d, self.cal_dim, np.eye(3), balance=0.5
        )

        # Use the principal point calculated by the model (no manual centering)
        
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.k, self.d, np.eye(3), new_k, self.cal_dim, cv2.CV_16SC2
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

def draw_lines(img, lines):
    line_img = np.zeros_like(img)
    overlay = img.copy()
    alpha = 0.5 
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

    def average_line(lines):
        if len(lines) == 0:
            return None
        x = []
        y = []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]
            y += [y1, y2]
        poly = np.polyfit(y, x, 1)
        return poly

    height = img.shape[0]
    y1 = height
    y2 = int(height * 0.65) # Shortened lines: only show close-range

    lane_coords = {'left': None, 'right': None}

    lane_coords = {'left': None, 'right': None}

    # Smoothing logic (Low Pass Filter) on poly coefficients
    for side, lines_list in [('left', left_lines), ('right', right_lines)]:
        poly = average_line(lines_list)
        
        # Apply LPF
        if poly is not None:
            if lane_poly_history[side] is not None:
                # NewPoly = Alpha * Current + (1-Alpha) * Last
                poly = LPF_ALPHA * poly + (1 - LPF_ALPHA) * lane_poly_history[side]
            lane_poly_history[side] = poly
        else:
            # If no line detected, use previous (optional: could decay)
            poly = lane_poly_history[side]

        # Draw if we have a valid poly (current or historical)
        if poly is not None:
            x1 = int(poly[0] * y1 + poly[1])
            x2 = int(poly[0] * y2 + poly[1])
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 6)
            lane_coords[side] = (x1, y1, x2, y2)

    # Perfect Masking: Only blend where lines are detected (preserves original contrast)
    line_mask = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY) > 0
    output = img.copy()
    # Apply alpha blending only to detected line pixels
    output[line_mask] = cv2.addWeighted(img, 0.4, line_img, 0.6, 0)[line_mask]
    
    return output, lane_coords['left'], lane_coords['right']

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

    # Initialize Lens Correction
    calibrator = CameraCalibrator(args.lens) if args.lens else None

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
        
        output, left_lane, right_lane = draw_lines(frame, hough_lines)
        
        # LDW
        if left_lane and right_lane:
            mid_bottom = ((left_lane[0]+right_lane[0])//2, left_lane[1])
            frame_center = w // 2
            deviation = int(frame_center - mid_bottom[0])

            if abs(deviation) > 50:
                cv2.putText(output, "LANE DEPARTURE WARNING!", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                cv2.line(output, (int(frame_center), int(h)), (int(mid_bottom[0]), int(mid_bottom[1])), (255,0,0), 3)
            else:
                cv2.putText(output, f"Deviation: {deviation}px", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

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
