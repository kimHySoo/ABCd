"""단일 다중 클래스 YOLO(53클래스, 색상+숫자를 직접 클래스로 탐지)를 mc/의 실제 카메라
홀드아웃 66장으로 평가해서, 2단계 파이프라인(YOLOv8n 탐지 + ResNet18 x2 분류)과 같은
기준(tile-multiset Precision/Recall/F1, 이미지별 완전일치)으로 직접 비교한다.

사용법: python eval_multiclass_yolo.py [가중치 경로] [출력 JSON 이름]
인자를 안 주면 ABCD_AI/rummikub의 best_yolo_2.pt를 기본값으로 평가한다.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

MC_ROOT = Path(r"C:\Users\SSAFY\workspace\mc")
ANNOTATION_DIR = MC_ROOT / "data" / "holdout" / "annotations"
DEFAULT_WEIGHTS_PATH = Path(r"C:\Users\SSAFY\workspace\ABCD_AI\rummikub\weights\tile_detector\best_yolo_2.pt")
CONF = 0.25
IMGSZ = 960


def canonicalize(raw_name: str) -> str:
    """'Black1'류(대문자+공백없음)와 'black_1'류(이미 표준 형식) 둘 다 처리한다."""
    if raw_name.lower() == "joker":
        return "joker"
    if raw_name.lower() in {f"{c}_{n}" for c in ("black", "blue", "orange", "red") for n in range(1, 14)}:
        return raw_name.lower()
    for color in ("black", "blue", "orange", "red"):
        if raw_name.lower().startswith(color):
            number = raw_name[len(color):].lstrip("_")
            if number.isdigit():
                return f"{color}_{int(number)}"
    raise ValueError(f"알 수 없는 클래스명: {raw_name}")


def load_ground_truth() -> dict[str, list[str]]:
    gt = {}
    for path in sorted(ANNOTATION_DIR.glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("reviewed"):
            continue
        stem = Path(rec["image"]).stem
        gt[stem] = [t["label"] for t in rec["tiles"]]
        gt_image_rel = rec["image"]
        gt.setdefault("_paths", {})
        gt["_paths"][stem] = gt_image_rel
    return gt


def main():
    weights_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS_PATH
    out_name = sys.argv[2] if len(sys.argv) > 2 else "multiclass_yolo_eval.json"

    gt = load_ground_truth()
    image_rel_paths = gt.pop("_paths")

    model = YOLO(str(weights_path))
    class_names = {i: canonicalize(n) for i, n in model.names.items()}

    total_matched = total_pred = total_gt = 0
    exact_match = 0
    n = 0
    total_ms = 0.0
    per_image_rows = []

    for stem, gt_labels in gt.items():
        image_path = MC_ROOT / "data" / image_rel_paths[stem]
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[스킵] 이미지 없음: {image_path}")
            continue

        t0 = time.perf_counter()
        result = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms

        pred_labels = [class_names[int(c)] for c in result.boxes.cls.cpu().numpy()]

        gt_counter = Counter(gt_labels)
        pred_counter = Counter(pred_labels)
        matched = sum((gt_counter & pred_counter).values())

        total_matched += matched
        total_pred += len(pred_labels)
        total_gt += len(gt_labels)
        n += 1
        is_exact = gt_counter == pred_counter
        if is_exact:
            exact_match += 1
        per_image_rows.append((stem, len(gt_labels), len(pred_labels), matched, is_exact))

    precision = total_matched / total_pred if total_pred else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"평가 이미지: {n}장")
    print(f"단일 다중 클래스 YOLO({weights_path.name}) -- Tile-multiset 기준")
    print(f"  Precision={precision:.1%}  Recall={recall:.1%}  F1={f1:.1%}")
    print(f"  이미지별 완전일치: {exact_match}/{n} ({exact_match/n:.1%})")
    print(f"  평균 추론 시간: {total_ms/n:.2f}ms/이미지 (이 PC의 GPU 기준, Jetson 아님 -- 상대 비교용)")

    print("\n불일치 이미지:")
    for stem, n_gt, n_pred, matched, is_exact in per_image_rows:
        if not is_exact:
            print(f"  {stem}: gt={n_gt} pred={n_pred} matched={matched}")

    out = {
        "model": f"{weights_path.name} (53-class single-stage)",
        "n": n, "precision": precision, "recall": recall, "f1": f1,
        "exact_match": exact_match / n if n else 0.0,
        "avg_ms": total_ms / n if n else 0.0,
    }
    out_path = MC_ROOT / "data" / out_name
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
