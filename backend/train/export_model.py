from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLO weights to ONNX")
    parser.add_argument("--weights", default="backend/models/best.pt", help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes")
    parser.add_argument("--half", action="store_true", help="Export FP16 model when supported")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights file not found: {weights}")

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
    )

    print("Export complete")
    print(f"Input weights : {weights.resolve()}")
    print(f"ONNX output   : {exported}")


if __name__ == "__main__":
    main()
