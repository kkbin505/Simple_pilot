#!/usr/bin/env python3
"""
Simple Pilot v2.0 - 改进版
包含：
  - LDW (Lane Departure Warning)
  - FCW (Forward Collision Warning)  
  - 焦距标定
  - 车道平滑滤波
  - 配置文件支持
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import logging
from collections import deque
from pathlib import Path

# ====== 日志配置 ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== 默认配置 ======
DEFAULT_CONFIG = {
    'lane_detection': {
        'canny_low': 50,
        'canny_high': 150,
        'hough_threshold': 50,
        'min_line_length': 40,
        'max_line_gap': 150,
        'roi_top_ratio': 0.6,
        'roi_left_ratio': 0.4,
        'roi_right_ratio': 0.6,
    },
    'ldw': {
        'deviation_threshold': 50,
        'enabled': True,
    },
    'fcw': {
        'focal_length': 920,        # 需要标定！
        'car_width_mm': 1800,
        'critical_distance': 1.5,
        'warning_distance': 2.5,
        'yolo_confidence': 0.4,
        'vehicle_classes': ['car', 'truck', 'bus', 'motorcycle'],
        'enabled': True,
    },
    'visualization': {
        'show_fps': True,
        'show_debug': True,
    }
}


# ====== 简单配置管理 ======
class Config:
    def __init__(self, data=None):
        self.data = data or DEFAULT_CONFIG.copy()
    
    def get(self, key, default=None):
        """获取配置值，支持 'key.subkey' 格式"""
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def set(self, key, value):
        """设置配置值"""
        keys = key.split('.')
        data = self.data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(config_dict)


# ====== 车道追踪器 ======
class LaneTracker:
    """使用历史平滑的车道追踪器"""
    
    def __init__(self, history_len=5):
        self.left_history = deque(maxlen=history_len)
        self.right_history = deque(maxlen=history_len)
        self.last_valid = None
    
    def update(self, left_lane, right_lane):
        """
        更新车道跟踪，返回平滑后的结果
        
        Args:
            left_lane: 左车道坐标 (x1, y1, x2, y2)
            right_lane: 右车道坐标 (x1, y1, x2, y2)
        
        Returns:
            平滑后的 (left_lane, right_lane)
        """
        if left_lane is not None and right_lane is not None:
            self.left_history.append(left_lane)
            self.right_history.append(right_lane)
            self.last_valid = (left_lane, right_lane)
        
        if not self.left_history or not self.right_history:
            return self.last_valid
        
        # 计算历史平均值
        try:
            left_avg = tuple(
                int(np.mean([l[i] for l in self.left_history]))
                for i in range(4)
            )
            right_avg = tuple(
                int(np.mean([r[i] for r in self.right_history]))
                for i in range(4)
            )
            return left_avg, right_avg
        except:
            return self.last_valid


# ====== 前车检测器 ======
class VehicleDetector:
    """YOLO 前车检测器"""
    
    def __init__(self, model_name='yolov8n.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
            logger.info(f"✓ YOLO loaded on {self.device}")
        except Exception as e:
            logger.error(f"✗ Failed to load YOLO: {e}")
            self.model = None
    
    def detect(self, frame, conf_threshold=0.4):
        """检测车辆"""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)[0]
            return results.boxes
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []


# ====== 距离估算器 ======
class DistanceEstimator:
    """基于焦距标定的距离估算器"""
    
    def __init__(self, focal_length=920, car_width_mm=1800):
        self.focal_length = focal_length
        self.car_width_mm = car_width_mm
        self.distance_history = deque(maxlen=3)
        self.calibration_data = []
    
    def estimate(self, bbox):
        """
        估算距离
        
        使用公式: distance = (car_width * focal_length) / bbox_width_px
        
        Args:
            bbox: 检测框 [x1, y1, x2, y2]
        
        Returns:
            距离 (米)
        """
        x1, y1, x2, y2 = map(int, bbox)
        bbox_width_px = x2 - x1
        
        if bbox_width_px < 20:  # 过小的检测框忽略
            return None
        
        # 距离计算
        distance_m = (self.car_width_mm * self.focal_length) / (bbox_width_px * 1000)
        
        # 历史平滑（3帧平均）
        self.distance_history.append(distance_m)
        smoothed = np.mean(list(self.distance_history))
        
        return smoothed
    
    def calibrate(self, distances_m, pixel_widths):
        """
        焦距标定
        
        Args:
            distances_m: 已知距离列表 [2.0, 3.0, 5.0]
            pixel_widths: 对应检测框像素宽度 [600, 400, 240]
        """
        if len(distances_m) != len(pixel_widths):
            logger.error("距离和像素宽度数量不匹配")
            return
        
        focal_lengths = [
            (self.car_width_mm * d * 1000) / w
            for d, w in zip(distances_m, pixel_widths)
        ]
        
        self.focal_length = int(np.mean(focal_lengths))
        
        logger.info("=== 焦距标定完成 ===")
        for d, w, f in zip(distances_m, pixel_widths, focal_lengths):
            logger.info(f"  距离 {d}m, 像素宽 {w}px → 焦距 {f:.1f}px")
        logger.info(f"平均焦距: {self.focal_length}px")
        
        return self.focal_length


# ====== 车道检测模块 ======
def get_roi_mask(img, config):
    """获取感兴趣区域 (ROI) 掩码"""
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    
    roi_top = int(height * config.get('lane_detection.roi_top_ratio'))
    roi_left = int(width * config.get('lane_detection.roi_left_ratio'))
    roi_right = int(width * config.get('lane_detection.roi_right_ratio'))
    
    polygon = np.array([[
        (0, height),
        (width, height),
        (roi_right, roi_top),
        (roi_left, roi_top),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    return mask


def detect_lane_lines(img, lines, config):
    """检测车道线并返回坐标"""
    if lines is None or len(lines) == 0:
        return None, None
    
    left_lines = []
    right_lines = []
    
    # 分离左右车道线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if x2 == x1:
            continue
        
        slope = (y2 - y1) / (x2 - x1)
        
        # 过滤不合理的斜率
        if abs(slope) < 0.5:
            continue
        
        if slope < 0:
            left_lines.append((x1, y1, x2, y2))
        else:
            right_lines.append((x1, y1, x2, y2))
    
    def fit_line(lines):
        """拟合直线"""
        if len(lines) == 0:
            return None
        
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None
        
        poly = np.polyfit(y_coords, x_coords, 1)  # 一次多项式
        return poly
    
    # 拟合左右车道线
    left_poly = fit_line(left_lines)
    right_poly = fit_line(right_lines)
    
    height = img.shape[0]
    y1, y2 = height, int(height * 0.6)
    
    left_lane = None
    right_lane = None
    
    if left_poly is not None:
        x1 = int(left_poly[0] * y1 + left_poly[1])
        x2 = int(left_poly[0] * y2 + left_poly[1])
        left_lane = (x1, y1, x2, y2)
    
    if right_poly is not None:
        x1 = int(right_poly[0] * y1 + right_poly[1])
        x2 = int(right_poly[0] * y2 + right_poly[1])
        right_lane = (x1, y1, x2, y2)
    
    return left_lane, right_lane


def detect_lanes(frame, config):
    """
    检测车道线
    
    Args:
        frame: 输入图像 (BGR)
        config: 配置对象
    
    Returns:
        (输出图像, 左车道, 右车道)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(
        blur,
        config.get('lane_detection.canny_low'),
        config.get('lane_detection.canny_high')
    )
    
    roi_mask = get_roi_mask(edges, config)
    roi = cv2.bitwise_and(edges, roi_mask)
    
    lines = cv2.HoughLinesP(
        roi, 1, np.pi/180,
        config.get('lane_detection.hough_threshold'),
        minLineLength=config.get('lane_detection.min_line_length'),
        maxLineGap=config.get('lane_detection.max_line_gap')
    )
    
    left_lane, right_lane = detect_lane_lines(frame, lines, config)
    
    # 绘制车道线
    output = frame.copy()
    if left_lane:
        x1, y1, x2, y2 = left_lane
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 6)
    
    if right_lane:
        x1, y1, x2, y2 = right_lane
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 6)
    
    return output, left_lane, right_lane


# ====== 可视化函数 ======
def draw_ldw_warning(img, left_lane, right_lane, deviation, threshold, config):
    """绘制 LDW 警告"""
    if not config.get('ldw.enabled'):
        return img
    
    h, w = img.shape[:2]
    
    if left_lane is None or right_lane is None:
        return img
    
    # 计算车道中心
    mid_bottom = ((left_lane[0] + right_lane[0]) // 2, left_lane[1])
    frame_center = w // 2
    
    # 绘制中心线
    cv2.line(img, (frame_center, h), (frame_center, h//2), (200, 200, 200), 1)
    
    # 绘制车道中心线
    cv2.line(img, (mid_bottom[0], h), (mid_bottom[0], h//2), (0, 255, 0), 2)
    
    # LDW 警告
    if abs(deviation) > threshold:
        cv2.rectangle(img, (0, 0), (w, 80), (0, 0, 255), -1)
        cv2.putText(img, "⚠️  LANE DEPARTURE WARNING!", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        cv2.putText(img, f"Deviation: {deviation:+d}px", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return img


def draw_fcw_warning(img, boxes, vehicle_detector, distance_estimator, config):
    """绘制 FCW 警告"""
    if not config.get('fcw.enabled') or vehicle_detector.model is None:
        return img, None, None
    
    h, w = img.shape[:2]
    
    warning_level = None
    closest_distance = float('inf')
    
    vehicle_classes = config.get('fcw.vehicle_classes', [])
    critical_dist = config.get('fcw.critical_distance')
    warning_dist = config.get('fcw.warning_distance')
    
    for box in boxes:
        try:
            cls_id = int(box.cls[0])
            label = vehicle_detector.model.names[cls_id]
            
            if label not in vehicle_classes:
                continue
            
            # 距离估算
            distance = distance_estimator.estimate(box.xyxy[0])
            
            if distance is None:
                continue
            
            closest_distance = min(closest_distance, distance)
            
            # 根据距离确定颜色
            if distance < critical_dist:
                color = (0, 0, 255)      # 红色
                warning_level = 'critical'
            elif distance < warning_dist:
                color = (0, 165, 255)    # 橙色
                warning_level = 'warning'
            else:
                color = (0, 255, 0)      # 绿色
            
            # 绘制检测框
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制距离标签
            label_text = f"{label} {distance:.2f}m"
            cv2.putText(img, label_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        except Exception as e:
            logger.debug(f"FCW error: {e}")
            continue
    
    # 绘制 FCW 警告
    if warning_level:
        if warning_level == 'critical':
            cv2.rectangle(img, (0, h-100), (w, h), (0, 0, 255), -1)
            cv2.putText(img, "🔴 CRITICAL COLLISION WARNING!", (20, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(img, (0, h-100), (w, h), (0, 165, 255), -1)
            cv2.putText(img, f"🟡 FORWARD COLLISION WARNING! {closest_distance:.2f}m", 
                       (20, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    return img, warning_level, closest_distance if closest_distance != float('inf') else None


# ====== 主函数 ======
def main(video_source=0, config=None):
    """
    主函数
    
    Args:
        video_source: 视频源 (0=摄像头, 或视频文件路径)
        config: 配置对象
    """
    if config is None:
        config = Config.from_dict(DEFAULT_CONFIG)
    
    # 初始化组件
    vehicle_detector = VehicleDetector()
    distance_estimator = DistanceEstimator(
        focal_length=config.get('fcw.focal_length'),
        car_width_mm=config.get('fcw.car_width_mm')
    )
    lane_tracker = LaneTracker(history_len=5)
    
    # 打开视频源
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video source: {video_source}")
        return False
    
    logger.info(f"✓ Video source opened")
    
    prev_time = 0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info(f"✓ Processing completed: {frame_count} frames")
                break
            
            frame_count += 1
            
            # ===== 车道检测 + LDW =====
            output, left_lane, right_lane = detect_lanes(frame, config)
            
            # 平滑处理
            left_lane, right_lane = lane_tracker.update(left_lane, right_lane)
            
            # 计算偏离度
            if left_lane and right_lane:
                h, w = output.shape[:2]
                mid_bottom = (left_lane[0] + right_lane[0]) // 2
                frame_center = w // 2
                deviation = frame_center - mid_bottom
            else:
                deviation = 0
            
            # 绘制 LDW
            output = draw_ldw_warning(output, left_lane, right_lane, 
                                     deviation, 
                                     config.get('ldw.deviation_threshold'),
                                     config)
            
            # ===== 前车检测 + FCW =====
            boxes = vehicle_detector.detect(frame, 
                                           config.get('fcw.yolo_confidence'))
            
            output, warning_level, closest_distance = draw_fcw_warning(
                output, boxes, vehicle_detector, distance_estimator, config
            )
            
            # ===== FPS 计算 =====
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            if config.get('visualization.show_fps'):
                cv2.putText(output, f"FPS: {fps:.1f}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # ===== 显示 =====
            cv2.imshow("Simple Pilot v2.0 - LDW + FCW", output)
            
            # 按 'q' 退出
            if cv2.waitKey(25) & 0xFF == ord('q'):
                logger.info("User exit")
                break
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        return False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return True


# ====== 焦距标定工具 =====
def calibrate_focal_length_interactive():
    """交互式焦距标定工具"""
    logger.info("=== 焦距标定工具 ===")
    logger.info("1. 准备一辆标准轿车（宽度约 1.8m）")
    logger.info("2. 使用摄像头在不同距离处拍摄")
    logger.info("3. 输入实际距离和对应的检测框像素宽度\n")
    
    distances_m = []
    pixel_widths = []
    
    while True:
        try:
            dist_input = input("距离 (m, 或 'done' 完成): ").strip()
            
            if dist_input.lower() == 'done':
                break
            
            distance = float(dist_input)
            pixel_width = float(input("检测框像素宽度 (px): "))
            
            distances_m.append(distance)
            pixel_widths.append(pixel_width)
            
            logger.info(f"✓ 已记录: {distance}m → {pixel_width}px\n")
        
        except ValueError:
            logger.error("✗ 输入格式错误，请重试\n")
            continue
    
    if len(distances_m) >= 2:
        estimator = DistanceEstimator()
        focal_length = estimator.calibrate(distances_m, pixel_widths)
        
        logger.info(f"\n将此值添加到配置中:")
        logger.info(f"  config.set('fcw.focal_length', {focal_length})")
        
        return focal_length
    else:
        logger.warning("✗ 需要至少 2 个标定点")
        return None


# ====== 命令行入口 ======
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Pilot v2.0 - LDW + FCW")
    parser.add_argument('--video', type=str, default='0',
                       help='Video file path (or 0 for webcam)')
    parser.add_argument('--calibrate', action='store_true',
                       help='Enter focal length calibration mode')
    parser.add_argument('--focal-length', type=int, default=920,
                       help='Focal length in pixels')
    parser.add_argument('--ldw', action='store_true', default=True,
                       help='Enable lane departure warning')
    parser.add_argument('--fcw', action='store_true', default=True,
                       help='Enable forward collision warning')
    
    args = parser.parse_args()
    
    # 焦距标定模式
    if args.calibrate:
        calibrate_focal_length_interactive()
    else:
        # 创建配置
        config = Config.from_dict(DEFAULT_CONFIG)
        config.set('fcw.focal_length', args.focal_length)
        config.set('ldw.enabled', args.ldw)
        config.set('fcw.enabled', args.fcw)
        
        # 解析视频源
        try:
            video_source = int(args.video)  # 摄像头 ID
        except ValueError:
            video_source = args.video  # 视频文件路径
        
        # 运行
        main(video_source, config)
