"""
Export YOLOv8n-seg to OpenVINO format optimized for Intel Iris Xe GPU
======================================================================
This script exports the YOLOv8n-seg model to OpenVINO IR format with FP16 precision,
specifically optimized for Intel 11th Gen integrated graphics (Iris Xe).

Usage:
    python export_openvino.py [--imgsz 640] [--model yolov8n-seg.pt]
"""

import argparse
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Install with: pip install ultralytics openvino-dev")
    sys.exit(1)


def export_model(model_path, imgsz=640):
    """
    Export YOLOv8 model to OpenVINO format.
    
    Args:
        model_path: Path to .pt model file
        imgsz: Input image size (default 640)
    """
    print(f"\n{'='*70}")
    print(f"Exporting YOLOv8-seg to OpenVINO IR Format")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Input Size: {imgsz}x{imgsz}")
    print(f"Precision: FP16 (Half Precision)")
    print(f"Target: Intel Iris Xe GPU")
    print(f"{'='*70}\n")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return False
    
    try:
        # Load model
        print("Loading model...")
        model = YOLO(model_path)
        
        # Export to OpenVINO
        print("Exporting to OpenVINO format...")
        print("This may take a few minutes...\n")
        
        export_path = model.export(
            format='openvino',
            imgsz=imgsz,
            half=True,  # FP16 precision for GPU acceleration
            simplify=True,  # Simplify ONNX graph
            dynamic=False,  # Static shape for better optimization
        )
        
        print(f"\n{'='*70}")
        print(f"✓ Export successful!")
        print(f"{'='*70}")
        print(f"Output directory: {export_path}")
        print(f"\nTo use this model, ensure the output directory exists and run:")
        print(f"  python simple_pilot_optimized.py")
        print(f"\nThe script will automatically detect and use the OpenVINO model.")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8n-seg to OpenVINO format for Intel Iris Xe GPU"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Path to YOLOv8 model file (default: yolov8n-seg.pt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640). Lower = faster, higher = more accurate"
    )
    
    args = parser.parse_args()
    
    # Validate image size
    if args.imgsz < 320 or args.imgsz > 1280:
        print("Warning: Recommended image size is between 320 and 1280")
    
    # Export model
    success = export_model(args.model, args.imgsz)
    
    if success:
        print("Next steps:")
        print("1. Run the optimized script:")
        print("   python simple_pilot_optimized.py")
        print("\n2. For video file processing:")
        print("   python simple_pilot_optimized.py path/to/video.mp4")
        print("\n3. For debugging:")
        print("   python simple_pilot_optimized.py --debug")
    else:
        print("\nTroubleshooting:")
        print("- Ensure ultralytics is installed: pip install ultralytics")
        print("- Ensure openvino-dev is installed: pip install openvino-dev")
        print("- Check that the model file exists")
        sys.exit(1)


if __name__ == "__main__":
    main()
