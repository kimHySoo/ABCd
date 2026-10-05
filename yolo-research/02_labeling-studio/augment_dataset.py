"""Augment C:\\Users\\SSAFY\\workspace\\rummikub_data with brightness and
horizontal-flip variants: 3 brightness levels (-25, 0, +25) x 2 flip states
(original, flipped) = 6 new images per source image, with YOLO labels
transformed to match (flip mirrors box x-coordinates; brightness doesn't
move anything so the box coordinates are copied as-is)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from brightness_ui import adjust_brightness
from evaluate_ui import RUMMIKUB_DATA_ROOT, SPLITS


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BRIGHTNESS_LEVELS = (-25, 0, 25)


def brightness_tag(value: int) -> str:
    if value > 0:
        return f"bp{value}"
    if value < 0:
        return f"bm{abs(value)}"
    return "b0"


AUGMENT_SUFFIXES = tuple(
    f"_{brightness_tag(level)}_{flip_tag}" for level in BRIGHTNESS_LEVELS for flip_tag in ("orig", "flip")
)


def is_augmented(stem: str) -> bool:
    """Recognizes files this script already produced, so re-running it on a
    folder that has grown with new originals doesn't augment its own output."""
    return stem.endswith(AUGMENT_SUFFIXES)


def flip_label_line(line: str) -> str:
    parts = line.split()
    class_id = parts[0]
    center_x, center_y, width, height = (float(value) for value in parts[1:5])
    center_x = 1.0 - center_x
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def build_variant(image_rgb, label_lines: list[str], brightness: int, flipped: bool):
    variant_image = adjust_brightness(image_rgb, brightness)
    variant_lines = label_lines
    if flipped:
        variant_image = cv2.flip(variant_image, 1)
        variant_lines = [flip_label_line(line) for line in label_lines]
    return variant_image, variant_lines


def augment_image(image_path: Path, label_path: Path, image_out_dir: Path, label_out_dir: Path) -> int:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  [경고] 이미지를 읽을 수 없습니다: {image_path.name}")
        return 0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    label_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    written = 0
    for brightness in BRIGHTNESS_LEVELS:
        for flipped in (False, True):
            variant_image, variant_lines = build_variant(image_rgb, label_lines, brightness, flipped)
            stem = f"{image_path.stem}_{brightness_tag(brightness)}_{'flip' if flipped else 'orig'}"
            out_image_path = image_out_dir / f"{stem}{image_path.suffix.lower()}"
            out_label_path = label_out_dir / f"{stem}.txt"
            variant_bgr = cv2.cvtColor(variant_image, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(out_image_path), variant_bgr):
                print(f"  [경고] 저장 실패: {out_image_path.name}")
                continue
            out_label_path.write_text(
                "\n".join(variant_lines) + ("\n" if variant_lines else ""), encoding="utf-8"
            )
            written += 1
    return written


def augment_split(root: Path, split: str) -> dict[str, int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    counts = {"originals": 0, "written": 0, "missing_label": 0, "unreadable": 0}
    if not image_dir.is_dir():
        return counts

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        if is_augmented(image_path.stem):
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            print(f"  [경고] 라벨 없음, 건너뜀: {image_path.name}")
            counts["missing_label"] += 1
            continue
        counts["originals"] += 1
        written = augment_image(image_path, label_path, image_dir, label_dir)
        if written == 0:
            counts["unreadable"] += 1
        counts["written"] += written
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augment rummikub_data with brightness (-25/0/+25) x flip (orig/flip) variants"
    )
    parser.add_argument("--root", type=Path, default=RUMMIKUB_DATA_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args()

    if not args.root.is_dir():
        raise FileNotFoundError(f"데이터 폴더가 없습니다: {args.root}")

    grand_total = {"originals": 0, "written": 0, "missing_label": 0, "unreadable": 0}
    for split in args.splits:
        print(f"[{split}]")
        counts = augment_split(args.root, split)
        for key, value in counts.items():
            grand_total[key] += value
        print(
            f"  원본 {counts['originals']}장 -> 증강 {counts['written']}장 저장 "
            f"(라벨 없음 {counts['missing_label']}장, 읽기 실패 {counts['unreadable']}장 건너뜀)"
        )

    print(
        f"\n전체: 원본 {grand_total['originals']}장 -> 증강 {grand_total['written']}장 저장 "
        f"(장당 최대 {len(BRIGHTNESS_LEVELS) * 2}장)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
