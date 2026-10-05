"""mc/data holdout 이미지 + YOLOv8s 초안 예측을, yolo_boardgame/review_labels_ui.py로
검수/수정할 수 있는 YOLO 포맷(images/<split>/*.jpg + labels/<split>/*.txt)으로 변환한다.

review_labels_ui.py는 순수 bbox+클래스만 다루는 도구다(그룹/판정 개념 없음) -- 이 변환의
목적은 "bbox 위치와 라벨(색상+숫자 또는 조커)이 맞는지"만 GUI로 빠르게 검수/수정하기
위함이다. 그룹/판정(decision)은 build_annotation_drafts.py가 만든
data/holdout/annotations/*.json에서 별도로 관리한다 -- bbox/라벨 교정이 끝나면 그 JSON도
다시 맞춰야 한다(추후 반영 스크립트 필요).

data/{init,middle,last} 폴더 구조를 review_labels_ui.py의 "split" 개념으로 그대로 쓴다.
클래스 인덱스는 yolo_boardgame/rummikub/classes.py의 CLASS_TO_ID를 그대로 재사용해서,
review_labels_ui.py가 기대하는 클래스 순서와 100% 맞춘다 (라벨 문자열 자체는
pipeline.py의 final_label과 같은 "color_number"/"joker" 규칙이라 별도 변환 불필요).

출력: data/holdout/yolo_review/images/<split>/*.jpg (원본 복사)
      data/holdout/yolo_review/labels/<split>/*.txt (YOLOv8s 예측 -> YOLO 포맷)
"""

import json
import shutil
import sys
from pathlib import Path

import cv2

PRED_DIR = Path("data/predictions")
DATA_DIR = Path("data")
OUT_ROOT = DATA_DIR / "holdout" / "yolo_review"
SPLITS = ("init", "middle", "last")

YOLO_BOARDGAME_ROOT = Path(r"C:\Users\SSAFY\workspace\yolo_boardgame")
sys.path.insert(0, str(YOLO_BOARDGAME_ROOT))

from rummikub.annotations import normalize_box  # noqa: E402
from rummikub.classes import CLASS_TO_ID  # noqa: E402

SOURCE_MODEL = "yolov8s"


def _latest_predictions(model_name: str) -> Path:
    files = sorted((PRED_DIR / model_name).glob("predictions_*.json"))
    if not files:
        raise SystemExit(f"{model_name}의 predictions_*.json이 없습니다. analyze_holdout.py를 먼저 실행하세요.")
    return files[-1]


if __name__ == "__main__":
    pred_path = _latest_predictions(SOURCE_MODEL)
    print(f"초안 기준: {pred_path}")
    records = json.loads(pred_path.read_text(encoding="utf-8"))

    skipped_labels: set[str] = set()
    skipped_existing = 0

    for record in records:
        image_rel = Path(record["image"])  # 예: init\capture_....jpg
        split = image_rel.parts[0]
        src_image_path = DATA_DIR / image_rel

        image_out_dir = OUT_ROOT / "images" / split
        label_out_dir = OUT_ROOT / "labels" / split
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        dst_image_path = image_out_dir / src_image_path.name
        if not dst_image_path.exists():
            shutil.copy2(src_image_path, dst_image_path)

        label_path = label_out_dir / f"{src_image_path.stem}.txt"
        if label_path.exists():
            # 이미 있으면 건드리지 않는다 -- review_labels_ui.py로 사람이 이미 고쳤을 수
            # 있어서, 재실행할 때마다 새 예측으로 덮어써버리면 수정 내용이 날아간다.
            skipped_existing += 1
            continue

        image = cv2.imread(str(src_image_path))
        if image is None:
            print(f"[skip] 읽기 실패: {src_image_path}")
            continue
        height, width = image.shape[:2]

        lines = []
        for detection in record["detections"]:
            label = detection["final_label"]
            if label not in CLASS_TO_ID:
                skipped_labels.add(label)
                continue
            box = {
                "class_id": CLASS_TO_ID[label],
                "x1": detection["x1"], "y1": detection["y1"],
                "x2": detection["x2"], "y2": detection["y2"],
            }
            try:
                lines.append(normalize_box(box, width, height))
            except ValueError:
                continue  # 너무 작은 박스 -- normalize_box가 자체적으로 거부

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if skipped_labels:
        print(f"[경고] CLASS_TO_ID에 없는 라벨이 있어 건너뜀: {skipped_labels}")
    if skipped_existing:
        print(f"[유지] 이미 라벨 파일이 있어 건드리지 않음: {skipped_existing}개 "
              "(다시 만들고 싶으면 해당 .txt를 직접 지우고 재실행하세요)")

    print(f"YOLO 포맷 변환 완료: {OUT_ROOT.resolve()}")
    for split in SPLITS:
        split_dir = OUT_ROOT / "images" / split
        n = len(list(split_dir.glob("*.jpg"))) if split_dir.is_dir() else 0
        print(f"  [{split}] {n}장")
