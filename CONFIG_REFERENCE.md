# Configuration Quick Reference

## 🎛️ Tunable Parameters in `simple_pilot_optimized.py`

All parameters are located at the top of the file in the `Configuration` section.

---

## Model & Inference

### `INFERENCE_SIZE`
- **Default**: `640`
- **Range**: `320` - `1280`
- **Effect**: Input resolution for YOLOv8 model
- **Trade-off**: Higher = more accurate but slower
- **Recommendations**:
  - `320`: Fast, good for testing (40-50 FPS)
  - `640`: **Recommended** balance (25-35 FPS)
  - `1280`: Best accuracy, slow (10-15 FPS)

### `CONF_THRESHOLD`
- **Default**: `0.35`
- **Range**: `0.1` - `0.9`
- **Effect**: Minimum confidence for detections
- **Trade-off**: Lower = more detections (more false positives)
- **Symptoms**:
  - Too low: Noisy detections, unstable lanes
  - Too high: Missed lanes, gaps in tracking
- **Tuning**: Start at 0.35, decrease if missing lanes, increase if too noisy

### `IOU_THRESHOLD`
- **Default**: `0.5`
- **Range**: `0.3` - `0.7`
- **Effect**: Non-Maximum Suppression threshold
- **Trade-off**: Lower = more aggressive duplicate removal
- **Recommendation**: Leave at 0.5 unless you see duplicate lane detections

---

## Motion Detection

### `MOTION_CHANGE_RATIO`
- **Default**: `0.008` (0.8%)
- **Range**: `0.001` - `0.05`
- **Effect**: Percentage of pixels that must change to detect motion
- **Symptoms**:
  - Too low: Never enters stationary mode (wastes power)
  - Too high: Enters stationary mode while moving slowly
- **Tuning**: Increase if it's too sensitive to camera shake

### `STATIONARY_FRAME_COUNT`
- **Default**: `15` frames (~0.5s at 30fps)
- **Range**: `5` - `60`
- **Effect**: Consecutive frames below threshold before declaring stationary
- **Trade-off**: Higher = slower to pause inference, fewer false triggers
- **Recommendation**: 15-30 frames for typical use

---

## Warning Thresholds

### `LDW_OFFSET_THRESHOLD_M`
- **Default**: `0.5` meters
- **Range**: `0.2` - `1.0`
- **Effect**: Lane departure warning trigger distance
- **Context**: US lane width ≈ 3.7m, so 0.5m ≈ 13% of lane width
- **Tuning**:
  - `0.3m`: Aggressive, early warnings
  - `0.5m`: **Recommended** balanced
  - `0.8m`: Relaxed, late warnings

### `FCW_DANGER_ZONE_Y`
- **Default**: `0.65` (bottom 35% of image)
- **Range**: `0.5` - `0.8`
- **Effect**: How close vehicle must be to trigger FCW
- **Trade-off**: Lower = earlier warnings (more false positives)
- **Context**: 0.65 ≈ 10-15m ahead at typical camera mounting

### `FCW_CENTER_X_MIN` / `FCW_CENTER_X_MAX`
- **Default**: `0.35` / `0.65` (center 30% of image)
- **Range**: `0.2` - `0.8`
- **Effect**: Horizontal zone for FCW (ignore vehicles in adjacent lanes)
- **Tuning**: Widen if you want warnings for adjacent lane vehicles

---

## Kalman Filter Tuning

### `KALMAN_PROCESS_NOISE`
- **Default**: `1e-4` (0.0001)
- **Range**: `1e-5` - `1e-2`
- **Effect**: How much we trust the model's predictions
- **Trade-off**: Lower = smoother but slower to adapt
- **Symptoms**:
  - Too low: Lanes drift over time, slow to respond to curves
  - Too high: Jittery lanes, defeats purpose of filtering
- **Tuning**: Decrease if lanes are too jittery

### `KALMAN_MEASUREMENT_NOISE`
- **Default**: `5e-2` (0.05)
- **Range**: `1e-3` - `1e-1`
- **Effect**: How much we trust the noisy detections
- **Trade-off**: Lower = trust measurements more (less smoothing)
- **Symptoms**:
  - Too low: Jittery lanes (follows noise)
  - Too high: Lanes lag behind actual position
- **Tuning**: Increase if lanes are too jittery, decrease if too laggy

### `KALMAN_DERIVATIVE_NOISE`
- **Default**: `1e-3` (0.001)
- **Range**: `1e-4` - `1e-2`
- **Effect**: How much we allow slope/intercept to change per frame
- **Trade-off**: Higher = faster adaptation to curves
- **Symptoms**:
  - Too low: Can't follow sharp curves
  - Too high: Unstable on straight roads
- **Tuning**: Increase for curvy roads, decrease for highways

---

## Kalman Filter Tuning Guide

### Scenario 1: Lanes are too jittery
```python
KALMAN_MEASUREMENT_NOISE = 1e-1  # Increase (trust measurements less)
KALMAN_PROCESS_NOISE = 1e-5      # Decrease (trust model more)
```

### Scenario 2: Lanes lag behind actual position
```python
KALMAN_MEASUREMENT_NOISE = 1e-2  # Decrease (trust measurements more)
KALMAN_DERIVATIVE_NOISE = 5e-3   # Increase (allow faster changes)
```

### Scenario 3: Can't follow curves
```python
KALMAN_DERIVATIVE_NOISE = 5e-3   # Increase
KALMAN_PROCESS_NOISE = 1e-3      # Increase
```

### Scenario 4: Lanes drift when detection is lost
```python
KALMAN_PROCESS_NOISE = 1e-5      # Decrease (hold position better)
# Also check: Are detections actually being lost? (use --debug mode)
```

---

## Geometry Calibration

### `LANE_WIDTH_M`
- **Default**: `3.7` meters (US standard)
- **Range**: `2.5` - `4.5`
- **Effect**: Used to convert pixel offset to meters
- **Calibration**: Measure actual lane width on your roads
- **Regional Standards**:
  - US: 3.7m (12 ft)
  - Europe: 3.5m
  - Narrow roads: 2.7-3.0m

### `CAMERA_FOCAL_LENGTH_PX`
- **Default**: `1000` pixels
- **Range**: `500` - `2000`
- **Effect**: Camera intrinsic parameter (currently unused, reserved for future FCW distance estimation)
- **Calibration**: Requires camera calibration procedure
- **Note**: Not critical for current implementation

---

## Visualization

### `SHOW_DEBUG_MASKS`
- **Default**: `False`
- **Effect**: Show segmentation mask in corner of display
- **Usage**: Set to `True` or toggle with 'd' key at runtime
- **Performance**: Minimal impact (~1-2 FPS)

### `LANE_COLOR_GOOD` / `LANE_COLOR_WARNING` / `LANE_COLOR_DANGER`
- **Default**: Green / Orange / Red
- **Format**: BGR tuple, e.g., `(0, 255, 0)`
- **Effect**: Color scheme for lane overlay
- **Customization**: Change to your preference

---

## Performance Tuning Recipes

### Maximum Speed (sacrifice accuracy)
```python
INFERENCE_SIZE = 320
CONF_THRESHOLD = 0.4
MOTION_CHANGE_RATIO = 0.01  # Less sensitive
```
**Expected**: 40-50 FPS

### Maximum Accuracy (sacrifice speed)
```python
INFERENCE_SIZE = 1280
CONF_THRESHOLD = 0.25
KALMAN_MEASUREMENT_NOISE = 1e-2
```
**Expected**: 10-15 FPS

### Balanced (recommended)
```python
INFERENCE_SIZE = 640
CONF_THRESHOLD = 0.35
KALMAN_MEASUREMENT_NOISE = 5e-2
```
**Expected**: 25-35 FPS

### Power Saving (for battery operation)
```python
INFERENCE_SIZE = 320
MOTION_CHANGE_RATIO = 0.005  # Very sensitive
STATIONARY_FRAME_COUNT = 10  # Quick to pause
```
**Expected**: Minimal GPU usage when stationary

---

## Testing Your Configuration

### 1. Visual Inspection
```bash
python simple_pilot_optimized.py --debug
```
- Watch for jittery lanes → Increase `KALMAN_MEASUREMENT_NOISE`
- Watch for laggy lanes → Decrease `KALMAN_MEASUREMENT_NOISE`
- Check tracker confidence in HUD → Should be > 0.7 for stable tracking

### 2. Performance Check
- Press 'd' to toggle debug mode
- Check "Inference FPS" in HUD
- Target: > 20 FPS for real-time use

### 3. Warning Accuracy
- Drive in center of lane → No LDW warning
- Drift toward edge → LDW warning should trigger at ~0.5m offset
- Approach vehicle → FCW warning when close

---

## Quick Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Low FPS | `INFERENCE_SIZE` too high | Reduce to 320 or 480 |
| Jittery lanes | `KALMAN_MEASUREMENT_NOISE` too low | Increase to 0.1 |
| Laggy lanes | `KALMAN_MEASUREMENT_NOISE` too high | Decrease to 0.01 |
| False LDW warnings | `LDW_OFFSET_THRESHOLD_M` too low | Increase to 0.7 |
| Missed LDW warnings | `LDW_OFFSET_THRESHOLD_M` too high | Decrease to 0.3 |
| No detections | `CONF_THRESHOLD` too high | Decrease to 0.25 |
| Noisy detections | `CONF_THRESHOLD` too low | Increase to 0.45 |
| Never goes stationary | `MOTION_CHANGE_RATIO` too low | Increase to 0.02 |
| Goes stationary while moving | `MOTION_CHANGE_RATIO` too high | Decrease to 0.005 |

---

## Advanced: Runtime Configuration (Future Feature)

Currently, all parameters require editing the source file. A future enhancement will add:
- JSON configuration file
- Command-line parameter overrides
- Real-time tuning UI

For now, create multiple copies with different configurations:
```bash
cp simple_pilot_optimized.py simple_pilot_highway.py  # High speed config
cp simple_pilot_optimized.py simple_pilot_city.py     # Low speed config
```

---

## 📚 See Also

- [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) - Detailed technical documentation
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Comparison with original version
- [README.md](README.md) - General usage instructions
