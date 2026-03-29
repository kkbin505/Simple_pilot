# Lane Detection Project - UFLDv2 ONNX

完整的基于 UFLDv2 ONNX 的车道线检测系统，CPU 实时运行。

## 🎯 项目位置

```
Simple_pilot/
└── lane_detection/          ← 新项目（独立）
    ├── main.py              # 主程序
    ├── model.py             # ONNX 推理
    ├── postprocess.py       # UFLDv2 解码
    ├── kalman.py            # 卡尔曼滤波
    ├── ipm.py               # 逆透视变换
    ├── visualize.py         # 可视化
    ├── video_reader.py      # 视频 I/O
    ├── utils.py             # 配置
    ├── requirements.txt     # 依赖
    ├── README.md            # 完整文档
    ├── QUICKSTART.md        # 快速开始
    ├── PROJECT_SUMMARY.md   # 项目总结
    └── ...                  # 其他辅助文件
```

## 🚀 快速开始

```bash
# 1. 进入项目目录
cd lane_detection

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载模型（查看说明）
python download_model.py

# 4. 运行检测
python main.py your_video.mp4
```

## 📚 详细文档

进入 `lane_detection/` 目录查看：
- **QUICKSTART.md** - 5 分钟快速上手
- **README.md** - 完整技术文档
- **PROJECT_SUMMARY.md** - 项目总结和使用指南

## ✅ 技术特性

- ✅ UFLDv2 ONNX 模型
- ✅ CPU ONLY（无需 GPU）
- ✅ 实时性能（≥20 FPS on i5）
- ✅ 模块化设计（8 个独立模块）
- ✅ 卡尔曼滤波平滑
- ✅ 完整可视化（HUD + 车道填充）
- ✅ 参数化配置
- ✅ 详细文档

## 📦 依赖

```
opencv-python
onnxruntime
numpy
scipy
```

## 🎯 使用示例

```bash
# 基本用法
python main.py input.mp4

# 指定输出
python main.py input.mp4 output.mp4

# 指定模型
python main.py input.mp4 output.mp4 path/to/model.onnx
```

## 📖 更多信息

查看 `lane_detection/` 目录中的详细文档。

---

**注意**: 这是一个独立项目，不会修改 `simple_pilot.py`。
