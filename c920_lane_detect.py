
import cv2
import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime
from threading import Thread, Lock
from queue import Queue

# Constants
DEFAULT_FPS = 30.0
WAIT_KEY_DELAY = 10 # ms

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
            # Note: We return the raw frame; the main thread needs to flip it for display
            # to match the recorded video orientation.
            if self.latest_frame is not None:
                return cv2.flip(self.latest_frame, -1)
            return None

    def stop(self):
        self.stopped = True
        self.capture_thread.join(timeout=1)
        self.writer_thread.join(timeout=2)
        self.cap.release()
        self.out.release()

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
        return img

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.5: continue
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))

    def average_line(lines):
        if len(lines) == 0: return None
        x, y = [], []
        for x1, y1, x2, y2 in lines:
            x += [x1, x2]; y += [y1, y2]
        try:
            return np.polyfit(y, x, 1)
        except:
            return None

    height = img.shape[0]
    y1, y2 = height, int(height * 0.6)

    for poly in [average_line(left_lines), average_line(right_lines)]:
        if poly is not None:
            try:
                x1, x2 = int(poly[0] * y1 + poly[1]), int(poly[0] * y2 + poly[1])
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 6)
            except: continue

    return cv2.addWeighted(img, 1.0, line_img, 1.0, 0)

def measure_camera_fps(camera_idx, duration=2.0):
    """Measure actual camera FPS before starting recording"""
    print(f"\nMeasuring camera FPS (please wait {duration}s)...")
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, _ = cap.read()
        if ret:
            frame_count += 1
    
    elapsed = time.time() - start_time
    cap.release()
    
    measured_fps = frame_count / elapsed
    print(f"Measured: {measured_fps:.1f} FPS")
    return measured_fps

def main():
    parser = argparse.ArgumentParser(description="Lane Detection with Auto-FPS Calibration")
    parser.add_argument("--camera", type=int, default=1, help="Camera Index")
    parser.add_argument("--fps", type=float, default=None, help="Recording FPS (auto if not set)")
    args = parser.parse_args()
    
    # Auto-detect FPS if not specified
    if args.fps is None:
        measured = measure_camera_fps(args.camera)
        if measured is not None:
            # Round to nearest common FPS value
            if measured > 25:
                args.fps = 30.0
            elif measured > 18:
                args.fps = 20.0
            elif measured > 12:
                args.fps = 15.0
            else:
                args.fps = 10.0
            print(f"Auto-selected recording FPS: {args.fps}")
        else:
            args.fps = DEFAULT_FPS
            print(f"Could not measure, using default: {args.fps}")
    else:
        print(f"Using manual FPS: {args.fps}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'output_{timestamp}.mp4'

    print(f"Initializing Buffered Stream on Camera {args.camera}...")
    
    try:
        stream = VideoStream(args.camera, 1280, 720, args.fps, output_filename)
        print(f"Codec: {stream.codec}")
        print(f"Recorded Video: {stream.width}x{stream.height} @ {args.fps} FPS")
    except Exception as e:
        print(f"Error starting stream: {e}")
        return

    stream.start()
    
    print("\nRecording started. Press 'q' to stop.")
    
    last_lines = None
    frame_count = 0
    prev_time = time.time()

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # Processing every 3rd frame
        # Only run heavy processing every 3 frames
        frame_count += 1
        if frame_count % 3 == 0:
            try:
                # 转换到 HSL 空间，对黄/白线更敏感
                hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
                
                # --- 白色过滤 --- (高亮度)
                lower_white = np.array([0, 200, 0])
                upper_white = np.array([180, 255, 255])
                white_mask = cv2.inRange(hls, lower_white, upper_white)
                
                # --- 黄色过滤 --- (特定色相 + 饱和度)
                lower_yellow = np.array([15, 30, 115])
                upper_yellow = np.array([35, 204, 255])
                yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
                
                # 合并掩码
                combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
                
                # 结合边缘检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                edges = cv2.Canny(blur, 50, 150)
                
                # 最终检测底图：边缘与颜色过滤结果取交集或并集
                # 这里我们用 bitwise_or，既保留边缘特征，又强制突出黄白区域
                lane_binary = cv2.bitwise_and(edges, combined_mask)
                
                roi = region_of_interest(lane_binary)
                lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=150)
                last_lines = lines # Update cache
            except:
                pass

        # Visualization
        display_frame = draw_lines(frame, last_lines)
        
        # Stats
        curr_time = time.time()
        display_fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        
        cv2.putText(display_frame, f"Display FPS: {display_fps:.1f}", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Capture FPS: {stream.actual_capture_fps:.1f}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Queue: {stream.write_queue.qsize()}", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Lane Detection (Buffered)', display_frame)

        if cv2.waitKey(WAIT_KEY_DELAY) & 0xFF == ord('q'):
            break

        if stream.stopped:
            break

    print("\nStopping stream and saving video...")
    stream.stop()
    cv2.destroyAllWindows()
    print("Capture complete.")

if __name__ == "__main__":
    main()
