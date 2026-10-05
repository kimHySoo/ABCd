"""analyze_holdout.py가 저장한 두 모델의 predictions.json을 이미지별로 비교해서,
판정(decision)이 갈린 이미지나 탐지 수 차이가 큰 이미지만 추려 보여주는 스크립트.

66장을 전부 눈으로 보기엔 시간이 많이 들어서, "두 모델이 실제로 다르게 본 지점"만
빠르게 좁혀서 annotated/ 이미지를 확인할 대상을 줄이는 용도다.
"""

import json
import sys
from pathlib import Path

PRED_DIR = Path("data/predictions")


def _latest(model_name: str) -> Path:
    files = sorted((PRED_DIR / model_name).glob("predictions_*.json"))
    if not files:
        raise SystemExit(f"{model_name}의 predictions_*.json이 없습니다.")
    return files[-1]


def load(model_name: str) -> dict[str, dict]:
    path = _latest(model_name)
    print(f"[{model_name}] {path.name}")
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["image"]: r for r in records}


if __name__ == "__main__":
    a_name, b_name = (sys.argv[1:3] if len(sys.argv) >= 3 else ("yolo26n", "yolov8s"))
    a = load(a_name)
    b = load(b_name)

    common = sorted(set(a) & set(b))
    print(f"\n공통 이미지 {len(common)}장 중 차이 나는 것만 표시\n")

    decision_diff = []
    detection_diff = []
    for image in common:
        ra, rb = a[image], b[image]
        if ra["decision"] != rb["decision"]:
            decision_diff.append(image)
        n_a, n_b = len(ra["detections"]), len(rb["detections"])
        if abs(n_a - n_b) >= 1:
            detection_diff.append((image, n_a, n_b))

    print(f"=== decision이 다른 이미지 ({len(decision_diff)}장) ===")
    for image in decision_diff:
        ra, rb = a[image], b[image]
        print(
            f"{image}: {a_name}={ra['decision']}{ra['reason_codes']} "
            f"vs {b_name}={rb['decision']}{rb['reason_codes']}"
        )

    print(f"\n=== 탐지 수가 2개 이상 차이나는 이미지 ({len(detection_diff)}장) ===")
    for image, n_a, n_b in detection_diff:
        print(f"{image}: {a_name}={n_a} vs {b_name}={n_b}")
