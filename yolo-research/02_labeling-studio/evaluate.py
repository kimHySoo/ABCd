"""Evaluate a Rummikub tile detector (mAP, precision, recall)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from rummikub.classes import CLASS_NAMES, canonicalize_external_name


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Rummikub tile detector")
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Low threshold so mAP is integrated over all candidate boxes (not the inference threshold)",
    )
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--name", default="rummikub_tiles_eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {args.weights}. "
            "Copy a trained best.pt into the project root or train with train.py."
        )

    model = YOLO(str(args.weights))

    model_names = [model.names[i] for i in sorted(model.names)]
    try:
        canonical_from_model = [canonicalize_external_name(name) for name in model_names]
    except ValueError:
        canonical_from_model = None
    if canonical_from_model != CLASS_NAMES:
        print(
            "[경고] 이 가중치의 클래스 순서가 dataset.yaml/rummikub.classes.CLASS_NAMES와 다릅니다.\n"
            "       precision/recall/mAP가 클래스별로 잘못 매칭되어 계산될 수 있습니다.\n"
            f"       모델 클래스 순서(0~4): {model_names[:5]}\n"
            f"       dataset.yaml 순서(0~4): {CLASS_NAMES[:5]}"
        )

    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        plots=True,
    )

    print(f"\n[{args.split}] {args.weights}")
    print(f"  Precision (mean): {metrics.box.mp:.3f}")
    print(f"  Recall (mean):    {metrics.box.mr:.3f}")
    print(f"  mAP50:            {metrics.box.map50:.3f}")
    print(f"  mAP50-95:         {metrics.box.map:.3f}")

    print("\nPer-class mAP50-95 (0.000 = no predictions or no ground truth in this split):")
    for class_id, class_map in enumerate(metrics.box.maps):
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        print(f"  {name:12s} {class_map:.3f}")

    print(f"\nPR curve / confusion matrix saved under runs/detect/{args.name}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
