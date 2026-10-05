"""Batch auto-label a folder of images with a trained YOLO model (default: best.pt)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from rummikub.annotations import DATA_ROOT, SPLITS, dataset_summary, save_yolo_sample
from rummikub.smart_label import local_yolo_auto_boxes


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch auto-label images with a trained Rummikub YOLO model")
    parser.add_argument("--source", type=Path, required=True, help="Folder of raw images to label")
    parser.add_argument("--weights", type=Path, default=ROOT / "best.pt")
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--allow-empty", action="store_true", help="Also save images with zero detections")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    if not args.source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    image_paths = sorted(
        p for p in args.source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    print(f"대상 이미지: {len(image_paths)}장 (가중치: {args.weights.name}, split: {args.split})")

    labeled = skipped = total_boxes = 0
    for path in image_paths:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            print(f"[{path.name}] 읽기 실패 (건너뜀)")
            skipped += 1
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes = local_yolo_auto_boxes(image_rgb, str(args.weights), args.confidence)
        if not boxes and not args.allow_empty:
            print(f"[{path.name}] 탐지 없음 (건너뜀, --allow-empty로 저장 가능)")
            skipped += 1
            continue
        image_path, _ = save_yolo_sample(image_rgb, boxes, args.split, DATA_ROOT)
        print(f"[{path.name}] 타일 {len(boxes)}개 -> {image_path.name}")
        labeled += 1
        total_boxes += len(boxes)

    print(f"\n라벨링 완료: {labeled}장 저장, {skipped}장 건너뜀, 총 박스 {total_boxes}개")
    print("자동 라벨은 정답이 아닙니다 — studio.py의 '데이터 라벨링' 탭 '기존 라벨 검수'에서")
    print("반드시 사람이 확인/수정한 뒤에 evaluate.py의 신뢰할 수 있는 정답으로 쓸 수 있습니다.")
    print()
    print(dataset_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
