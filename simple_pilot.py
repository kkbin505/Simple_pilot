import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import time
import argparse

# Fix for OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# prev_time = 0  <- moved inside main or managed globally
width_ratio = 0.6
width_extention = -50
left_offset = 0
LPF_ALPHA = 0.15  # Smoothing factor: 1.0 = no filter, 0.05 = hyper smooth/slow
lane_poly_history = {'left': None, 'right': None}

# ====== Front Vehicle Detection ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("yolov8n.pt")  # Official Ultralytics YOLOv8

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# ====== Lane Detection and LDW Functions ======
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[(-width_extention+left_offset, height), (width+width_extention, height), 
                         (int(width*width_ratio), int(height * 0.6)), (int(width*(1-width_ratio))+left_offset, int(height * 0.6))]])
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
    y2 = int(height * 0.6)

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
    args = parser.parse_args()

    # Determine if source is video file or camera index
    source = args.video_source
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open source: {source}")
        return

    # Seek to start time if specified
    if args.start > 0:
        # print(f"⏭️ Skipping to {args.start} seconds...")
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    prev_time = 0  # <- Initialize FPS timer
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        h, w = frame.shape[:2]

        # ----- Lane Detection + LDW -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = region_of_interest(edges)
        
        hough_lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=60, maxLineGap=150)
        output, left_lane, right_lane = draw_lines(frame, hough_lines)
        # cv2.imshow("Simple Pilot - LDW + FCW", output)
        # Debug console output
        # if hough_lines is not None:
        #    print(f"DEBUG: Found {len(hough_lines)} raw line segments")
        # else:
        #    print("DEBUG: No line segments detected")
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
            # print(f"Lanes detected! Center deviation: {deviation}px")
        else:
            # print("Warning: One or both lane lines NOT detected")
            pass

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


        cv2.imshow("Simple Pilot - LDW + FCW", output)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
