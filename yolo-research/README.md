# yolo-research

ABCd 프로젝트에서 루미큐브 타일 인식을 위해 직접 진행한 YOLO 관련 실험 코드와 노트북 모음입니다. 진행 순서대로 폴더를 나눴습니다.

```text
01 웹캠·전처리 실험 ──▶ 02 데이터 구축 도구 ──▶ 03 YOLO 학습 ──▶ 04 실제 카메라 평가
 (07/22~07/23)           (07/23~08/07)           (07/30~08/07)       (08/07~08/11)
```

**포함하지 않은 것**

- 학습 가중치(`*.pt`), 촬영 원본 이미지·라벨 데이터, 가상환경 — 용량과 촬영 환경 프라이버시 때문에 제외했습니다.
- 프로젝트 본체 코드(mc의 `pipeline.py`, `rules.py`, `mqtt_handler.py` 등) — SSAFY 반출 신청이 필요한 부분이라 제외했습니다. 그래서 본체를 import 하는 스크립트(탐지기 4종을 홀드아웃에 돌리는 `analyze_holdout.py`, TensorRT 벤치마크 등)도 함께 빠졌습니다.
- ResNet18 / MobileNetV3 분류기만 다루는 학습 노트북.

코드 안의 경로는 원래 작업 환경(`C:\Users\SSAFY\workspace\...`, 외부 GPU 서버의 `rummikub_0730/`) 기준이라, 그대로 실행하려면 경로와 데이터를 다시 맞춰야 합니다.

---

## 01_webcam-preprocessing

원본 위치: `rummikub-webcam/`. 공개 데이터로 학습된 53클래스 YOLO 가중치로 실제 카메라 조건을 확인하고, 보드를 위에서 본 구도로 펴는 전처리를 실험했습니다.

| 파일 | 내용 |
| --- | --- |
| `webcam_detect.py` | 웹캠 실시간 YOLO 탐지 (사용 가능한 카메라 자동 탐색, 화면 버튼으로 카메라 전환) |
| `panorama.py` | 웹캠으로 여러 장을 찍어 `cv2.Stitcher`로 파노라마 합성 — 넓은 테이블을 한 장에 담을 수 있는지 확인 |
| `test.ipynb` | iPhone 사진에 YOLO 탐지 적용 |
| `hief_to_jpeg.ipynb` | iPhone HEIC 사진을 JPEG로 일괄 변환 |
| `holography.ipynb` | A4 네 모서리 초록 테이프를 HSV로 검출 → 호모그래피로 A4 평면화 |
| `test4.ipynb` | 원본 vs A4 평면화 이미지의 YOLO 탐지 결과 비교 |
| `test7.ipynb` | 85×110cm 게임 테이블 평면화 + 원본/평면화 탐지 비교 |
| `test8.ipynb` | 45×35.5cm 영역 crop → 평면화, 흰 책상 vs 검정 매트 배경별 탐지 성능 비교 |
| `visualize_bboxes.ipynb` | YOLO 라벨 파일의 bbox를 이미지 위에 그려 라벨 확인 |
| `HOMOGRAPHY.md` | 평면화 노트북 환경 구성과 동작 개요 |

**결론**: 평면화는 라벨링 구도를 일정하게 만드는 데는 유용했지만, 카메라를 고정 각도로 거치하면 평면화 없이도 탐지 성능이 충분해서 실시간 추론 파이프라인에서는 빼고 **라벨링 단계에서만** 사용했습니다.

## 02_labeling-studio

원본 위치: `yolo_boardgame/`. 데이터를 빠르게 만들고 정답을 검수하기 위한 Gradio 도구 모음입니다. 자세한 사용법은 폴더 안 [`README.md`](02_labeling-studio/README.md), 작업 기록은 [`docs/log.md`](02_labeling-studio/docs/log.md)에 있습니다.

| 파일 | 내용 |
| --- | --- |
| `studio.py` | 라벨링 스튜디오: MobileSAM 클릭 한 번 박스, 수동 박스, 자동 라벨 초안, 턴 규칙 검증 탭 |
| `import_roboflow.py` | Roboflow Universe 공개 라벨(1,255장, 13,065 박스)을 YOLO Detect 형식으로 변환 |
| `auto_label.py` | 폴더 단위로 `best.pt` 자동 라벨 초안 생성 (사람 검수 전제) |
| `evaluate_ui.py` | 이미지 1장 탐지 → 표에서 삭제/수정/누락 추가 → 학습 데이터로 저장 |
| `evaluate_ui_batch.py` | 위 흐름을 폴더 단위로 순차 진행 |
| `evaluate_ui_flatten.py` | 폴더 단위 + 테이프 자동 검출 평탄화 후 검수 |
| `evaluate_ui_manual_flatten.py` | 테이프 검출이 실패할 때 모서리 4점을 직접 클릭해 평탄화 |
| `review_labels_ui.py` | 이미 있는 이미지+라벨 쌍을 불러와 검수·수정 (홀드아웃 66장 정답 검수에 사용) |
| `review_groups_ui.py` | 타일 그룹(멜드) 경계 정답 검수 |
| `augment_dataset.py` | 밝기(-25/0/+25) × 좌우반전으로 장당 6배 증강, 라벨 좌표 함께 변환 |
| `brightness_ui.py` | HSV 명도만 조정하는 밝기 미리보기 |
| `validate_dataset.py` | 이미지-라벨 쌍, 클래스 번호, 좌표 범위, 클래스 순서 일치 검사 |
| `train.py` / `evaluate.py` | `best.pt` 기반 파인튜닝, Ultralytics mAP/정밀도/재현율 평가 |
| `rummikub/` | 클래스 정의, 라벨 변환, 탐지기 래퍼, MobileSAM 스마트 라벨, 규칙 검증 모듈 |
| `tests/` | 라벨 변환·규칙 검증 단위 테스트 |
| `docs/` | 데이터 출처(CC BY 4.0), 규칙과 한계, 작업 로그 |

**겪은 문제**: 기존 `best.pt`의 클래스 순서가 숫자순이 아니라 Roboflow 원본 라벨의 **알파벳순**(`Black1, Black10, Black11, …`)이라, 그대로 평가하면 클래스별 지표가 엉뚱하게 매칭됐습니다. `CLASS_NAMES`와 `dataset.yaml`을 같은 순서로 맞추고 `validate_dataset.py`에 순서 불일치 검사를 넣었습니다.

## 03_yolo-training

원본 위치: 워크스페이스 루트. 외부 GPU 서버(NVIDIA L40S)에서 실행한 학습 노트북입니다.

| 파일 | 내용 |
| --- | --- |
| `train_models.ipynb` | 첫 구조: 53클래스(색+숫자 조합) YOLO26 탐지기 + ResNet18 숫자/색상 분류기 학습 |
| `evaluate_models.ipynb` | 위 세 모델을 각 test split으로 평가 |
| `train_yolo_tile_only.ipynb` | 색·숫자는 분류기에 맡기고 YOLO는 **1클래스 `tile` 위치 탐지**만 하도록 전환 (YOLOv8s) |
| `train_yolo_tile_augmented.ipynb` | 회전 4 × 밝기 2 = 장당 8배 증강, **원본 단위로 먼저 7:2:1 분할** 후 증강해 누수 방지, YOLOv8s / YOLO26n 비교 |
| `train_yolo_tile_yolov8n.ipynb` | 같은 데이터·시드·설정으로 YOLOv8n 학습 (3-way 비교) |
| `train_yolo_tile_yolo11n.ipynb` | 같은 조건으로 YOLO11n 학습 (4-way 비교) |
| `infer_test_images_8v.ipynb` | 라벨 없는 이미지에 YOLOv8s + ResNet18 ×2 2단계 추론 데모 |

**겪은 문제**: 1클래스로 바꿨는데도 탐지가 전혀 안 됐습니다. 원인은 데이터셋을 symlink로 만들었을 때 `Path.resolve()`가 링크를 원본 경로로 따라가, Ultralytics가 새 1클래스 라벨이 아닌 원본 53클래스 라벨을 읽은 것이었습니다. 실제 복사로 바꾸고 라벨이 전부 class 0인지 전수 검사하는 셀을 추가했습니다.

**학습 결과** (증강 test 944장, 타일 12,824개, seed 42 / epochs 100 / imgsz 960)

| 모델 | Precision | Recall | mAP50 | mAP50-95 | 파라미터 | GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s | 0.998 | 0.997 | 0.995 | 0.854 | 11.1M | 28.4 |
| YOLOv8n | 0.996 | 0.997 | 0.995 | 0.853 | 3.01M | 8.1 |
| YOLO11n | 0.999 | 0.998 | 0.995 | 0.854 | 2.58M | 6.3 |
| YOLO26n | 0.996 | 0.996 | 0.995 | 0.849 | 2.4M | 5.2 |

네 모델이 사실상 동률이라 이 지표만으로는 고를 수 없었고, 그래서 04의 실제 카메라 평가가 필요했습니다.

## 04_detector-evaluation

원본 위치: `mc/`. Jetson 실제 촬영 경로로 찍은 홀드아웃 66장(PASS 52 / 반칙 14)에 사람이 검수한 정답을 만들고, 탐지기를 비교한 스크립트입니다.

| 파일 | 내용 |
| --- | --- |
| `build_yolo_review_dataset.py` | 홀드아웃 이미지 + 모델 초안 예측을 `review_labels_ui.py`가 읽는 YOLO 형식으로 변환 (02의 `rummikub/classes.py` 사용) |
| `score_tiles_only.py` | 그룹핑·판정을 빼고 탐지+분류 결과(타일 multiset)만 정답과 비교 — 탐지기 비교의 기준 지표 |
| `score_against_annotations.py` | 모델별 최종 PASS/FAIL 정확도와 confusion matrix |
| `diff_predictions.py` | 두 모델의 예측을 이미지별로 비교해 판정이 갈린 이미지만 추림 |
| `eval_multiclass_yolo.py` | 단일 53클래스 YOLO를 2단계 구조와 같은 기준으로 평가 |
| `results/` | 단일 53클래스 YOLO 체크포인트 2종의 평가 결과 JSON |

**실제 카메라 66장 결과**

| 구성 | 타일 F1 | 타일 완전일치 |
| --- | ---: | ---: |
| **YOLOv8n + ResNet18 ×2 (채택)** | 99.9% | **97.0% (64/66)** |
| YOLOv8s + ResNet18 ×2 | — | 93.9% (62/66) |
| YOLO11n + ResNet18 ×2 | — | 93.9% (62/66) |
| YOLO26n + ResNet18 ×2 | 99.5% | 80.3% (53/66) |
| 단일 53클래스 YOLO ① / ② | 74.4% / 62.3% | 0/66 |

증강 데이터에서는 동률이던 모델들이 실제 카메라에서는 크게 갈렸습니다. 이 결과로 프로덕션에 먼저 넣었던 YOLO26n을 되돌리고 YOLOv8n으로 확정했습니다. 단일 53클래스 체크포인트와의 비교는 학습 조건이 통제된 ablation이 아니라서, "이 프로젝트의 데이터와 조건에서 일관되게 낮았다"까지만 주장합니다.
