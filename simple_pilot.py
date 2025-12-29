import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time


VIDEO_PATH = "D:\LiZhen\Github\Simple_pilot\demo\yolo.mov"

prev_time = 0

# ====== 前车检测 ======
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("yolov8n.pt")  # ultralytics 官方 YOLOv8

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']

# ====== 车道检测和 LDW 函数（与之前类似） ======
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.4), int(height * 0.6)),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines):
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

    for side, poly in zip(['left', 'right'], [average_line(left_lines), average_line(right_lines)]):
        if poly is None:
            continue
        x1 = int(poly[0] * y1 + poly[1])
        x2 = int(poly[0] * y2 + poly[1])
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 6)
        lane_coords[side] = (x1, y1, x2, y2)

    output = cv2.addWeighted(img, 1.0, line_img, 1.0, 0)
    return output, lane_coords['left'], lane_coords['right']

# ====== 主循环 ======
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return
    prev_time = 0  # <- 初始化 FPS 计时
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # ----- lane detection + LDW -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = region_of_interest(edges)
        output, left_lane, right_lane = draw_lines(frame, cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=150))


        h, w, _ = frame.shape

        # ----- Lane Detection + LDW -----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = region_of_interest(edges)
        output, left_lane, right_lane = draw_lines(frame, cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=150))

        # LDW
        if left_lane and right_lane:
            mid_bottom = ((left_lane[0]+right_lane[0])//2, left_lane[1])
            frame_center = w//2
            deviation = frame_center - mid_bottom[0]

            if abs(deviation) > 50:
                cv2.putText(output, "LANE DEPARTURE WARNING!", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                cv2.line(output, (frame_center, h), (mid_bottom[0], mid_bottom[1]), (0,0,255), 3)
            else:
                cv2.putText(output, f"Deviation: {deviation}px", (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # ----- YOLO 前车检测 + FCW -----
        results = yolo_model(frame, conf=0.4)[0]

        warning_fcw = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_height_ratio = (y2 - y1) / h

            if box_height_ratio > 0.35:  # 简单距离近似
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
        prev_time = curr_time  # <- 更新上次时间

        cv2.putText(output, f"FPS: {fps:.1f}", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


        cv2.imshow("Simple Pilot - LDW + FCW", output)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
