"""data/holdout/annotations/*.json(정답, review_labels_ui.py로 사람이 검수)와
data/predictions/<model>/predictions_*.json(모델 예측)을 대조해서, 각 모델의 실제
PASS/FAIL 정확도와 confusion matrix를 계산한다.

EXPERIMENT_PLAN.md 4.5절의 "최종 판정" 지표(PASS/FAIL accuracy, 반칙 precision/recall)를
가장 단순한 형태로 먼저 낸다 -- 탐지/분류 단계별 IoU 매칭(4.4절 전체 스펙)까지는 아직
아니고, "이 이미지가 실제로 PASS인데 모델도 PASS라고 했는가"만 본다. 마감이 촉박해서
가장 핵심적인 지표(최종 판정 정확도)부터 확정하기 위함이다.
"""

import json
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


def load_ground_truth() -> dict[str, str]:
    truth = {}
    for path in ANNOTATION_DIR.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("reviewed"):
            continue
        truth[Path(record["image"]).stem] = record["decision"]
    return truth


if __name__ == "__main__":
    truth = load_ground_truth()
    print(f"정답(reviewed=true) {len(truth)}장 기준\n")

    for model_name in MODELS:
        pred_path = _latest(model_name)
        if pred_path is None:
            print(f"[{model_name}] 예측 파일 없음, 건너뜀")
            continue

        records = json.loads(pred_path.read_text(encoding="utf-8"))
        tp = tn = fp = fn = 0  # PASS=positive 기준: FP=실제 FAIL인데 PASS로 놓침(false acceptance)
        missing = 0

        for record in records:
            stem = Path(record["image"]).stem
            if stem not in truth:
                missing += 1
                continue
            true_decision = truth[stem]
            pred_decision = record["decision"]

            if true_decision == "PASS" and pred_decision == "PASS":
                tp += 1
            elif true_decision == "FAIL" and pred_decision == "FAIL":
                tn += 1
            elif true_decision == "FAIL" and pred_decision == "PASS":
                fp += 1  # false acceptance: 반칙을 놓침
            elif true_decision == "PASS" and pred_decision == "FAIL":
                fn += 1  # false rejection: 정상을 막음

        n = tp + tn + fp + fn
        accuracy = (tp + tn) / n if n else 0.0
        false_acceptance_rate = fp / (fp + tn) if (fp + tn) else 0.0  # 실제 FAIL 중 놓친 비율
        false_rejection_rate = fn / (fn + tp) if (fn + tp) else 0.0  # 실제 PASS 중 막은 비율

        print(f"=== {model_name} ({pred_path.name}) ===")
        print(f"  n={n} (정답과 매칭 안 된 이미지 {missing}장 제외)")
        print(f"  PASS/FAIL accuracy: {accuracy:.1%}  (TP={tp} TN={tn} FP={fp} FN={fn})")
        # 분모가 작으면(특히 실제 FAIL 표본 수) 0%가 "절대 안 놓침"이 아니라 "이 작은
        # 표본에서는 안 놓침"이라는 뜻이라 분모를 항상 같이 보고한다.
        print(f"  false acceptance rate (반칙을 PASS로 놓침): {false_acceptance_rate:.1%} "
              f"({fp}/{fp + tn})")
        print(f"  false rejection rate (정상을 FAIL로 막음): {false_rejection_rate:.1%} "
              f"({fn}/{fn + tp})")
        print()
