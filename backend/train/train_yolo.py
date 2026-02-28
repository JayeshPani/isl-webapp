from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO model for ISL recognition")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device: cpu, 0, 0,1, etc.")
    parser.add_argument("--project", default="runs/isl", help="Ultralytics project output folder")
    parser.add_argument("--name", default="yolo_train", help="Run name")
    return parser.parse_args()


def load_model_with_fallback(primary_model: str) -> Tuple["YOLO", str]:
    from ultralytics import YOLO

    tried: List[str] = []
    candidates = [primary_model]
    if primary_model == "yolo11n.pt":
        candidates.extend(["yolo8n.pt", "yolov8n.pt"])

    last_error: Exception | None = None
    for candidate in candidates:
        tried.append(candidate)
        try:
            return YOLO(candidate), candidate
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    tried_text = ", ".join(tried)
    raise RuntimeError(f"Could not load model from candidates: {tried_text}") from last_error


def resolve_best_path(model: "YOLO", train_result: object, project: str, name: str) -> Path:
    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        best = getattr(trainer, "best", None)
        if best:
            return Path(str(best))

    save_dir = getattr(train_result, "save_dir", None)
    if save_dir:
        return Path(str(save_dir)) / "weights" / "best.pt"

    return Path(project) / name / "weights" / "best.pt"


def main() -> None:
    args = parse_args()

    try:
        model, used_model = load_model_with_fallback(args.model)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Model load failed: {exc}")

    print(f"[INFO] Starting training with model: {used_model}")
    print(f"[INFO] Dataset config: {args.data}")

    train_result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best_path = resolve_best_path(model, train_result, args.project, args.name)

    print("\n========== Training Summary ==========")
    print(f"Run name      : {args.name}")
    print(f"Project path  : {Path(args.project).resolve()}")
    print(f"Base model    : {used_model}")
    print(f"Best weights  : {best_path}")
    print("======================================")


if __name__ == "__main__":
    main()
