"""그룹핑/판정(rules) 단계를 완전히 제외하고, 순수 탐지+분류 결과(타일 multiset)만
정답과 비교하는 스크립트.

score_against_annotations.py의 PASS/FAIL 정확도는 그룹핑 알고리즘(MST)의 오류까지
같이 섞여서 나온다는 게 확인됐다(capture_20260802_211544 사례 -- 그룹핑이 잘못
병합한 걸 사람이 못 걸러내서 정답 자체가 한 번 틀렸었음). 탐지기/분류기 자체의
성능만 보려면 그룹/판정과 무관하게 "이 이미지에서 어떤 타일들을 찾았는가"만
비교해야 한다 -- bbox IoU 매칭까지는 안 하고, 같은 라벨이 몇 장씩 있는지(multiset)만
비교하는 단순화된 버전이다(4.4~4.5절의 정식 IoU 매칭보다 거칠지만, 그룹핑 오류의
영향은 완전히 제거된다).
"""

import json
from collections import Counter
from pathlib import Path

ANNOTATION_DIR = Path("data/holdout/annotations")
PRED_DIR = Path("data/predictions")
MODELS = (
    "yolo26n", "yolov8s", "yolov8n", "yolo11n",
    "resnet_separate", "resnet_multihead", "mobilenet_separate",
    "best_combo",
)


def _latest(model_name: str) -> Path | None:
    files = sorted((PRED_DIR / model_name).glob("predictions_*.json"))
    return files[-1] if files else None


def load_ground_truth() -> dict[str, list[str]]:
    truth = {}
    for path in ANNOTATION_DIR.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("reviewed"):
            continue
        truth[Path(record["image"]).stem] = [t["label"] for t in record["tiles"]]
    return truth


if __name__ == "__main__":
    truth = load_ground_truth()
    print(f"정답(reviewed=true) {len(truth)}장 기준 -- 그룹핑/판정 제외, 타일 multiset만 비교\n")

    for model_name in MODELS:
        pred_path = _latest(model_name)
        if pred_path is None:
            print(f"[{model_name}] 예측 파일 없음, 건너뜀")
            continue

        records = json.loads(pred_path.read_text(encoding="utf-8"))

        exact_match = 0
        total_tp = total_fp = total_fn = 0
        n = 0

        for record in records:
            stem = Path(record["image"]).stem
            if stem not in truth:
                continue
            n += 1

            truth_labels = Counter(truth[stem])
            pred_labels = Counter(d["final_label"] for d in record["detections"])

            if truth_labels == pred_labels:
                exact_match += 1

            all_labels = set(truth_labels) | set(pred_labels)
            for label in all_labels:
                t_count = truth_labels.get(label, 0)
                p_count = pred_labels.get(label, 0)
                total_tp += min(t_count, p_count)
                total_fp += max(0, p_count - t_count)
                total_fn += max(0, t_count - p_count)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print(f"=== {model_name} ({pred_path.name}) ===")
        print(f"  n={n}, 이미지별 타일 multiset 완전일치: {exact_match}/{n} ({exact_match/n:.1%})")
        print(f"  타일 단위 TP={total_tp} FP={total_fp} FN={total_fn}")
        # 주의: bbox IoU 매칭 없이 라벨 개수(Counter)만 비교한 것이라 엄밀한 객체 탐지
        # Precision/Recall이 아니다 -- 같은 라벨이 정답/예측에 각각 여러 개 있으면 위치가
        # 달라도 매칭된 것으로 센다. "Tile-multiset" 지표로 이름을 명확히 구분한다.
        print(f"  Tile-multiset Precision={precision:.1%}  "
              f"Tile-multiset Recall={recall:.1%}  Tile-multiset F1={f1:.1%}")
        print()
