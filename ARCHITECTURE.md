# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VIDEO INPUT                                     │
│                                                                              │
│                    ┌──────────────────────────────┐                         │
│                    │  Camera (C920)               │                         │
│                    │  1280x720 @ 30fps            │                         │
│                    │  Focus: Infinity Lock        │                         │
│                    └──────────┬───────────────────┘                         │
│                               │                                              │
└───────────────────────────────┼──────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THREAD ARCHITECTURE                                  │
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │ Capture Thread   │      │  Main Thread     │      │  Writer Thread   │  │
│  │                  │      │                  │      │                  │  │
│  │ • Read frames    │─────▶│ • Read latest    │      │ • Pop from queue │  │
│  │ • Queue for write│      │ • Run inference  │      │ • Flip & write   │  │
│  │ • Update latest  │      │ • Draw overlay   │      │ • Save to MP4    │  │
│  │                  │      │                  │      │                  │  │
│  │ Lock-free read ──┼──────┤ Lock-protected   │      │ Queue-based      │  │
│  └──────────────────┘      └──────────────────┘      └──────────────────┘  │
│                                     │                                        │
└─────────────────────────────────────┼────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING PIPELINE                                  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 1. MOTION DETECTION                                                  │  │
│  │    • Downscale to 320x180                                            │  │
│  │    • Frame differencing                                              │  │
│  │    • Change ratio > 0.8%?                                            │  │
│  └────────────────────┬─────────────────────────────────────────────────┘  │
│                       │                                                     │
│              ┌────────┴────────┐                                            │
│              │                 │                                            │
│         YES  ▼                 ▼  NO                                        │
│  ┌─────────────────┐   ┌─────────────────┐                                │
│  │ FULL INFERENCE  │   │ PREDICTION ONLY │                                │
│  └────────┬────────┘   └────────┬────────┘                                │
│           │                     │                                           │
│           ▼                     │                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 2. YOLOv8n-seg INFERENCE (OpenVINO FP16)                             │  │
│  │    • Input: 640x640                                                  │  │
│  │    • Device: Intel Iris Xe GPU                                       │  │
│  │    • Precision: FP16 (half precision)                                │  │
│  │    • Output: Segmentation masks + Bounding boxes                     │  │
│  └────────────────────┬─────────────────────────────────────────────────┘  │
│                       │                                                     │
│                       ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 3. MASK PROCESSING                                                   │  │
│  │    • Extract semantic masks                                          │  │
│  │    • Separate left/right by centroid                                 │  │
│  │    • Filter by vertical position (ignore sky)                        │  │
│  └────────────────────┬─────────────────────────────────────────────────┘  │
│                       │                                                     │
│                       ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 4. WEIGHTED LINE FITTING                                             │  │
│  │    • Weight = (y / height)²  (bottom-heavy)                          │  │
│  │    • Fit: x = my + b                                                 │  │
│  │    • Output: (m_left, b_left), (m_right, b_right)                    │  │
│  └────────────────────┬─────────────────────────────────────────────────┘  │
│                       │                                                     │
│           ┌───────────┴───────────┐                                         │
│           │                       │                                         │
│           ▼                       ▼                                         │
│  ┌─────────────────┐     ┌─────────────────┐                              │
│  │ 5a. KALMAN      │     │ 5b. KALMAN      │                              │
│  │     TRACKER     │     │     TRACKER     │                              │
│  │     (LEFT)      │     │     (RIGHT)     │                              │
│  │                 │     │                 │                              │
│  │ State: [m,b,    │     │ State: [m,b,    │                              │
│  │         dm,db]  │     │         dm,db]  │                              │
│  │                 │     │                 │                              │
│  │ • Predict()     │     │ • Predict()     │                              │
│  │ • Update(m,b)   │     │ • Update(m,b)   │                              │
│  │ • Confidence    │     │ • Confidence    │                              │
│  └────────┬────────┘     └────────┬────────┘                              │
│           │                       │                                         │
│           └───────────┬───────────┘                                         │
│                       │                                                     │
│                       ▼                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 6. WARNING LOGIC                                                     │  │
│  │                                                                      │  │
│  │    LDW (Lane Departure Warning):                                    │  │
│  │    • Calculate lane center: (x_left + x_right) / 2                  │  │
│  │    • Offset = lane_center - image_center                            │  │
│  │    • Convert to meters: offset_m = offset_px / lane_width_px * 3.7m │  │
│  │    • Trigger if |offset_m| > 0.5m                                   │  │
│  │                                                                      │  │
│  │    FCW (Forward Collision Warning):                                 │  │
│  │    • Check bounding boxes in danger zone (bottom 35%)               │  │
│  │    • Filter by class: car, motorcycle, bus, truck                   │  │
│  │    • Trigger if vehicle in center lane area                         │  │
│  └────────────────────┬─────────────────────────────────────────────────┘  │
│                       │                                                     │
└───────────────────────┼─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VISUALIZATION & OUTPUT                               │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 7. DRAW OVERLAY                                                      │  │
│  │    • Lane lines (color-coded: green/orange/red)                      │  │
│  │    • Filled lane area (semi-transparent)                             │  │
│  │    • HUD: Status, FPS, Offset, Confidence                            │  │
│  │    • Warnings: LDW/FCW text + border                                 │  │
│  │    • Debug: Segmentation mask (if enabled)                           │  │
│  └────────────────────┬─────────────────────────────────────────────────┘  │
│                       │                                                     │
│              ┌────────┴────────┐                                            │
│              │                 │                                            │
│              ▼                 ▼                                            │
│  ┌─────────────────┐   ┌─────────────────┐                                │
│  │   CV2 DISPLAY   │   │   MP4 WRITER    │                                │
│  │   (Real-time)   │   │   (Recorded)    │                                │
│  └─────────────────┘   └─────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              PERFORMANCE METRICS
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────┬──────────────┬──────────────┬─────────────────┐
│ Stage               │ Latency      │ Frequency    │ Notes           │
├─────────────────────┼──────────────┼──────────────┼─────────────────┤
│ Motion Detection    │ 2ms          │ Every frame  │ Downscaled      │
│ YOLOv8 Inference    │ 30ms         │ If moving    │ OpenVINO FP16   │
│ Mask Processing     │ 5ms          │ If moving    │ NumPy ops       │
│ Weighted Fitting    │ 3ms          │ If moving    │ Polyfit         │
│ Kalman Update       │ 0.5ms        │ Every frame  │ Matrix ops      │
│ Warning Logic       │ 1ms          │ Every frame  │ Simple math     │
│ Visualization       │ 7ms          │ Every frame  │ OpenCV drawing  │
├─────────────────────┼──────────────┼──────────────┼─────────────────┤
│ TOTAL (Moving)      │ 48.5ms       │ ~20 FPS      │ Full pipeline   │
│ TOTAL (Stationary)  │ 10.5ms       │ ~95 FPS      │ Inference off   │
└─────────────────────┴──────────────┴──────────────┴─────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Camera → Capture Thread → [Queue] → Writer Thread → MP4 File
           │
           └─→ [Latest Frame] → Main Thread → Processing Pipeline → Display


═══════════════════════════════════════════════════════════════════════════════
                           KALMAN FILTER DETAILS
═══════════════════════════════════════════════════════════════════════════════

State Vector (4D):
  x = [m, b, dm, db]ᵀ
  
  m  = slope (x = my + b)
  b  = intercept
  dm = rate of change of slope
  db = rate of change of intercept

Transition Model (Constant Velocity):
  ┌   ┐   ┌           ┐ ┌   ┐
  │ m │   │ 1 0 1 0   │ │ m │
  │ b │ = │ 0 1 0 1   │ │ b │
  │dm │   │ 0 0 1 0   │ │dm │
  │db │   │ 0 0 0 1   │ │db │
  └   ┘   └           ┘ └   ┘

Measurement Model:
  z = [m, b]ᵀ  (directly measure slope and intercept)

Noise Covariance:
  Process Noise (Q):  1e-4 (trust model)
  Measurement Noise (R): 5e-2 (moderate trust in detections)


═══════════════════════════════════════════════════════════════════════════════
                           CONFIGURATION SUMMARY
═══════════════════════════════════════════════════════════════════════════════

Model:
  • YOLOv8n-seg (6.5MB)
  • Input: 640x640
  • Format: OpenVINO IR (FP16)
  • Device: Intel Iris Xe GPU

Thresholds:
  • Confidence: 0.35
  • IoU (NMS): 0.5
  • Motion: 0.8% pixel change
  • LDW: 0.5m offset
  • FCW: Bottom 35% + center 30%

Kalman Filter:
  • Process Noise: 1e-4
  • Measurement Noise: 5e-2
  • Derivative Noise: 1e-3

Performance:
  • Target FPS: 20-30
  • Latency: <50ms
  • Power Saving: 70% when stationary


═══════════════════════════════════════════════════════════════════════════════
```
