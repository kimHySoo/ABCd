# 프로젝트 파일 구성 및 .gitignore 정리

`yolo_boardgame`(Rummikub YOLO26 Studio)의 전체 파일 용도와, `.gitignore`로
제외되는 항목 중 실제로 로컬에 준비/추가해야 하는 것을 정리한 문서입니다.
아직 Git 저장소로 초기화되어 있지 않으므로(`git init` 전), 실제 커밋 전에
아래 내용을 참고해 필요한 파일을 채워 넣으세요.

## 1. 최상위 파일

| 파일 | 용도 |
|---|---|
| `README.md` | 프로젝트 개요, 환경 구성, 라벨링/학습/턴 검증 사용법 |
| `.gitignore` | Git 추적 제외 규칙 (아래 3절 참고) |
| `requirements.txt` | 메인 프로젝트 의존성 (`ultralytics`, `gradio`, `opencv-python`, `matplotlib`, `ipykernel` 등). `rummikub-webcam`과 공유하는 `abcd` 가상환경이라 그 프로젝트 노트북이 쓰는 패키지도 포함 |
| `dataset.yaml` | YOLO 학습용 데이터셋 설정. 53개 클래스(색상 1~13 + `joker`) 정의. 클래스 순서는 `rummikub/classes.py`의 `CLASS_NAMES`(= `best.pt` 학습 순서)와 반드시 동일해야 함 |
| `setup.ps1` | `uv`로 `abcd` 가상환경(Python 3.10) 생성, PyTorch(CUDA 13.0)/의존성 설치, `boardgame-ai` Jupyter 커널 등록, GPU 확인 스크립트 |
| `studio.py` | Gradio 기반 로컬 스튜디오 — 라벨링, 턴 룰 검증, 학습 안내 UI |
| `import_roboflow.py` | Roboflow Universe 공개 라벨(`rummikub-p8akb`)을 로컬 YOLO 형식으로 가져오는 스크립트 |
| `auto_label.py` | 폴더 단위로 `best.pt`가 이미지를 일괄 자동 라벨링해 `data/images`·`data/labels`에 저장하는 배치 스크립트 (`studio.py`의 대화형 자동 라벨링과 같은 로직 재사용). 결과는 사람 검수 전까지는 정답으로 쓸 수 없음 |
| `best.pt` | 기본 모델 가중치. 루미큐브 타일로 이미 학습된 체크포인트(`rummikub-webcam` 프로젝트에서 복사해옴). 저장소에는 포함되지 않으므로(`.gitignore`의 `*.pt`) 직접 복사해야 함 (3.1절 참고) |
| `train.py` | `best.pt`를 시작점으로 로컬 데이터를 추가 파인튜닝하는 스크립트 (더 이상 COCO 사전학습 `yolo26n.pt`부터 새로 학습하지 않음) |
| `validate_dataset.py` | `data/images`·`data/labels`의 이미지-라벨 쌍, 클래스 번호, 좌표 범위 검사 + `dataset.yaml`이 `CLASS_NAMES`와 같은 클래스 순서인지 검사 (라벨 형식만 검사, 모델 성능은 측정 안 함) |
| `evaluate.py` | `best.pt`(또는 학습 결과)를 `dataset.yaml` split(train/val/test)으로 평가 — 정밀도, 재현율, mAP50, mAP50-95, 클래스별 mAP50-95 출력. 가중치의 클래스 순서가 `rummikub.classes.CLASS_NAMES`와 다르면 경고 출력 |
| `evaluate_ui.py` | 이미지 1장을 올리면 `best.pt`로 탐지 → 박스별 표에서 중복/오탐 행 삭제·클래스 수정 → 검수된 결과를 `C:\Users\SSAFY\workspace\rummikub_data`에 YOLO 데이터셋(이미지+txt)으로 저장하는 간단한 Gradio 페이지. `studio.py`보다 가볍게 "탐지 확인 + 즉석 교정" 용도 |
| `evaluate_ui_batch.py` | `evaluate_ui.py`와 같은 기능이지만 폴더 경로를 입력하면 그 안의 이미지를 순서대로 하나씩 불러오고, 저장할 때마다 자동으로 다음 이미지로 넘어가는 폴더 일괄 검수 페이지. `evaluate_ui.py`의 함수를 그대로 import해서 재사용 |
| `evaluate_ui_flatten.py` | `evaluate_ui_batch.py`와 같은 폴더 일괄 흐름에 파란 테이프(청테이프) 모서리 자동 인식 + 실측 가로/세로 비율 평탄화 단계를 추가한 페이지. `evaluate_ui.py`의 검수/저장 함수를 재사용 |
| `brightness_ui.py` | YOLO/탐지와 무관한 독립 페이지. 이미지 업로드 후 슬라이더로 밝기(HSV V값) 조정, 실시간 미리보기 |
| `augment_dataset.py` | `rummikub_data`의 이미지를 밝기(-25/0/+25) x 좌우반전 조합으로 6장씩 증강하고 YOLO 라벨도 함께 변환. 재실행해도 이미 증강된 파일은 다시 증강하지 않음 |
| `review_labels_ui.py` | `rummikub_data_2`처럼 **이미 라벨링된** 이미지+txt 쌍을 폴더 단위로 불러와 박스를 겹쳐 보여주고 검수/수정하는 페이지. 모델 탐지는 하지 않으며, 저장하면 새 파일을 만들지 않고 원본 라벨 txt를 그 자리에서 덮어씀. 박스 편집 도구는 `evaluate_ui.py`의 함수를 재사용 |

## 2. 디렉터리

| 경로 | 용도 |
|---|---|
| `docs/` | 데이터 출처(`DATA_PROVENANCE.md`), 규칙/한계(`RULES_AND_LIMITATIONS.md`) 등 문서 |
| `rummikub/` | 핵심 패키지 — `classes.py`(클래스 정의), `annotations.py`(YOLO 라벨 저장/시각화), `detector.py`(YOLO26 추론), `rules.py`(멜드/턴 판정 로직), `smart_label.py`(Roboflow/로컬 YOLO26/MobileSAM 자동 라벨) |
| `tests/` | `rummikub.annotations`, `rummikub.rules` 등에 대한 단위 테스트 |
| `legacy_roboflow/` | 이전 Roboflow 전용 CLI(`app.py`, `webcam.py`). 새 시스템에서는 사용하지 않고 참고용으로 보존 |
| `data/` | 학습 데이터 루트. 현재는 `roboflow_import_manifest.json`(가져오기 기록)만 있고, 이미지/라벨은 `.gitignore`로 제외되어 로컬에서 직접 채워야 함 (3절 참고) |

## 3. `.gitignore`에서 제외되는 항목과 내가 추가해야 하는 것

`.gitignore`는 아직 존재하지 않는 항목까지 미리 규칙으로 잡아둔 상태입니다.
현재 리포지토리에는 아래 패턴에 해당하는 파일이 하나도 실제로 존재하지
않으므로, **직접 만들거나 생성 스크립트를 실행해서 채워야** 합니다.

### 3.1 반드시 추가해야 하는 것 (없으면 프로젝트가 동작하지 않음)

- **`.env` (루트)** — `ROBOFLOW_API_KEY`가 필요합니다.
  `import_roboflow.py`와 `rummikub/smart_label.py`가 `./.env` 또는
  `legacy_roboflow/.env`에서 `ROBOFLOW_API_KEY=`를 직접 읽습니다.
  - ⚠️ 루트에는 `legacy_roboflow/.env.example` 같은 템플릿이 없습니다.
    `legacy_roboflow/.env.example`을 참고해 루트에 `.env`를 새로 만들거나,
    루트용 `.env.example`을 추가해두는 것을 권장합니다.
  - 값 예시:
    ```
    ROBOFLOW_API_KEY=본인의_private_api_key
    ROBOFLOW_MODEL_ID=rummikub-p8akb/2016
    ROBOFLOW_API_URL=https://serverless.roboflow.com
    ```
- **`legacy_roboflow/.env`** — `legacy_roboflow/app.py`, `webcam.py` 실행 시 필요.
  `legacy_roboflow/.env.example`을 복사해서 API 키만 채우면 됩니다.
  ```powershell
  Copy-Item legacy_roboflow\.env.example legacy_roboflow\.env
  ```
- **`abcd/`** — 가상환경 폴더(이름은 `abcd`). `setup.ps1` 실행으로 생성
  (`uv`가 Python 3.10을 자동으로 받아옴 + CUDA 13.0 PyTorch). `rummikub-webcam`과
  공유하는 환경이며, `setup.ps1`이 `boardgame-ai`라는 이름의 Jupyter 커널로도
  등록해서 `rummikub-webcam`의 노트북들이 이 가상환경을 그대로 씁니다.
- **`data/images/{train,val,test}/`, `data/labels/{train,val,test}/`** — 실제
  라벨링 데이터. `import_roboflow.py` 실행(공개 라벨 가져오기) 또는
  `studio.py`의 라벨링 탭에서 직접 채워야 학습/검증이 가능합니다.
- **`best.pt` (루트)** — `*.pt`는 `.gitignore`로 제외되어 저장소에 포함되지 않으므로,
  다른 프로젝트(`rummikub-webcam`)에서 학습된 가중치를 직접 복사해 넣어야 합니다(이미 완료됨).
  이 체크포인트는 Roboflow 원본 라벨 텍스트(`Black1`, `Black10`, ..., `Joker`, ...)를
  알파벳순으로 정렬한 클래스 순서로 학습되었습니다. `rummikub/classes.py`의
  `CLASS_NAMES`와 `dataset.yaml`을 이미 이 순서에 맞춰뒀으므로 새로 저장되는 라벨은
  자동으로 맞습니다. `evaluate.py`/`train.py`는 실행할 때마다 `best.pt`에 저장된
  실제 클래스 순서를 다시 확인해 어긋나면 경고를 출력합니다(예: 다른 가중치 파일로
  바꿨을 때).

### 3.2 스크립트 실행 결과로 자동 생성되는 것 (직접 만들 필요 없음)

- **`runs/`** — `train.py` 학습 결과와 `evaluate.py` 평가 결과. 가중치는
  `runs/detect/rummikub_tiles/weights/best.pt`(및 `last.pt`)에, 평가용 PR 곡선/혼동
  행렬은 `runs/detect/rummikub_tiles_eval/`에 저장됨.
- **`weights/`, `*.onnx` 등 추가 변환 산출물** — 필요 시 내보내기 결과.
  (루트의 `best.pt` 자체는 위 3.1절처럼 직접 복사해 넣는 파일이며 자동 생성되지 않음)
- **`outputs/`** — 추론/검출 결과 출력.
- **`data/turns/`** — 스튜디오의 "턴 룰 검증" 탭에서 저장하는 기준/현재 턴 이미지.
- **`__pycache__/`, `.pytest_cache/`, `.coverage` 등** — Python/테스트 캐시.
- **`build/`, `dist/`, `*.egg-info/`** — 패키징 시 생성.

### 3.3 필요할 때만 만들면 되는 것

- **`.idea/`, `.vscode/`** — 개인 IDE 설정 (선택).
- **`tmp/`, `temp/`, `*.tmp`, `*.log`** — 임시 파일/로그 (자동 발생 시 무시용).

## 4. 새 데이터셋 라벨링 (`rummikub_data_2`)

`evaluate_ui.py`(및 이를 재사용하는 `evaluate_ui_batch.py`, `evaluate_ui_flatten.py`)의
저장 경로(`RUMMIKUB_DATA_ROOT`)를 `C:\Users\SSAFY\workspace\rummikub_data`에서
`C:\Users\SSAFY\workspace\rummikub_data_2`로 변경했습니다. 세 페이지 모두
`evaluate_ui.py`의 `RUMMIKUB_DATA_ROOT`를 그대로 import해서 쓰므로, 이 한 곳만
바꾸면 셋 다 새 경로에 저장됩니다. (`augment_dataset.py --root`의 기본값도 같은
상수를 참조하므로 함께 `rummikub_data_2`를 가리키게 됩니다.)

### `rummikub_data_2`에 저장되는 페이지 3개 (URL 정리)

| # | 페이지 | 실행 명령 | URL | 용도 |
|---|---|---|---|---|
| 1 | `evaluate_ui.py` | `.\abcd\Scripts\python.exe evaluate_ui.py` | http://127.0.0.1:7861 | 이미지 1장 업로드 → (선택)자르기/확대 → `best.pt` 탐지 → 표에서 중복 삭제·클래스 수정·누락 박스 추가 → 저장 |
| 2 | `evaluate_ui_batch.py` | `.\abcd\Scripts\python.exe evaluate_ui_batch.py` | http://127.0.0.1:7862 | 위와 동일한 흐름이지만 폴더 경로를 입력하면 안의 이미지를 순서대로 불러오고, 저장할 때마다 자동으로 다음 이미지로 넘어감 |
| 3 | `evaluate_ui_flatten.py` | `.\abcd\Scripts\python.exe evaluate_ui_flatten.py` | http://127.0.0.1:7863 | 폴더 일괄 처리 + 초록 테이프 모서리 인식으로 원근 보정(평탄화) 후 동일한 검수/저장 흐름 |
| 4 | `review_labels_ui.py` | `.\abcd\Scripts\python.exe review_labels_ui.py` | http://127.0.0.1:7865 | 새로 탐지하지 않고, **이미 있는** 이미지+라벨 쌍을 폴더/split 단위로 불러와 검수. 저장하면 원본 라벨 txt를 덮어씀 |
| 5 | `evaluate_ui_manual_flatten.py` | `.\abcd\Scripts\python.exe evaluate_ui_manual_flatten.py` | http://127.0.0.1:7866 | `evaluate_ui_flatten.py`와 같은 폴더 일괄 흐름이지만, 테이프 자동 검출 대신 **이미지 위에서 테이블 모서리 4개를 직접 클릭**해 평탄화. 자동 검출이 실패하거나(타일에 가려짐, 테이프 없음/색 안 맞음) 잘못된 모서리를 잡을 때 사용 |

앞의 3개(1~3)와 5번 페이지는 새 샘플을 만들어 `evaluate_ui.py`의 `RUMMIKUB_DATA_ROOT` 상수를 공유해서
`C:\Users\SSAFY\workspace\rummikub_data_2`에 저장합니다. 실행 시
`--server-port`로 포트를, `--no-browser`로 브라우저 자동 실행을 끌 수 있습니다
(예: `.\abcd\Scripts\python.exe evaluate_ui_batch.py --server-port 7862 --no-browser`).

> 참고: `studio.py`(http://127.0.0.1:7860)는 별도의 라벨링/턴 검증 통합
> 페이지로, 이번에 변경한 `rummikub_data_2`가 아니라 프로젝트 내 `data/`
> (`rummikub/annotations.py`의 `DATA_ROOT`)에 저장하므로 이 3개 URL에는
> 포함하지 않았습니다.

### `evaluate_ui_flatten.py`의 평탄화 = `ABCD_AI` 프로덕션 캘리브레이션과 동일

`evaluate_ui_flatten.py`의 평탄화 로직을 `C:\Users\SSAFY\workspace\ABCD_AI\rummikub`의
`app/services/calibration_service.py`(`CalibrationService`)와
`test/pipeline_crop_save.ipynb`의 테이프 제거 방식과 동일하게 맞췄습니다. 라벨링용
크롭 이미지가 실제 서비스가 YOLO에 넣는 이미지와 같은 방식으로 만들어져야 학습
효과가 있기 때문입니다.

- **테이프 색**: 파란 테이프 → **초록 테이프**로 변경 (`GREEN_HSV_LOWER/UPPER`와
  동일한 Hue 40~85, S/V 하한 60). UI의 "초록 테이프 Hue 하한/상한" 슬라이더 기본값도
  이에 맞춰졌습니다.
- **모서리 검출**: 최소 넓이 200px² 이상인 컨투어 중 가장 큰 4개를 초록 테이프로 사용
  (기존의 이미지 크기 비율 기반 최소/최대 넓이 필터 대신, `MIN_TAPE_AREA_PX`와 동일한
  절대 픽셀 기준).
- **모서리 정렬**: 각도 기반 정렬 대신 `CalibrationService.order_corners`와 동일한
  x+y / x-y 기반 정렬(좌상=합 최소, 우하=합 최대, 우상=차 최대, 좌하=차 최소) 사용.
- **출력 크기**: 측정된 모서리 간 픽셀 거리로 매번 다르게 정하던 방식 대신, "테이블
  가로/세로 길이(cm) × 배율(px/cm)"로 정해지는 **고정 캔버스 크기**로 통일
  (기본값 110cm × 85cm × 10px/cm = 1100×850px, `TABLE_SIZE_CM`/`PX_PER_CM`과 동일).
  회전된 촬영에도 가로/세로를 바꿔치기하지 않는 이유는 `order_corners`가 이미
  좌상/우상/우하/좌하를 회전에 무관하게 정확히 잡아주기 때문입니다.
- **테이프 자국 제거**: 평탄화 직후 남아있는 초록 테이프 자국을 `pipeline_crop_save.ipynb`의
  "캘리브레이션 테이프 제거" 셀과 동일하게 마스킹 후 보드의 중앙값 색으로 채워 지웁니다
  (탐지 단계에서 테이프가 타일로 오검출되는 것을 방지).

UI의 "1. 평탄화" 섹션에서 테이블 가로/세로 길이(cm)와 배율(px/cm)을 조정할 수 있고,
기본값이 이미 프로덕션 값(110/85/10)으로 맞춰져 있어 그대로 사용하면 됩니다.

### `evaluate_ui_manual_flatten.py` = 모서리 수동 선택 평탄화 (`test0729/test.ipynb` 방식)

`evaluate_ui_flatten.py`는 초록 테이프를 **자동으로** 찾아 평탄화하지만, 테이프가 타일에
가려지거나 없거나 색 범위를 벗어나면 자동 검출이 실패하거나 모서리를 잘못 잡습니다. 이럴
때 쓰는 페이지가 `evaluate_ui_manual_flatten.py`이며,
`ABCD_AI/rummikub/test/test0729/test.ipynb`의 수동 모서리 선택 도구(`QuadPicker` /
`order_quad` / `flatten_table(quad=...)`)와 동일한 방식으로 동작합니다.

- **모서리 선택**: 이미지를 **좌상(TL) → 우상(TR) → 우하(BR) → 좌하(BL)** 순서로 4번
  클릭. "마지막 점 취소"/"모서리 초기화"로 다시 찍을 수 있음 (노트북의 좌클릭 추가/우클릭
  취소를 버튼으로 구현).
- **정렬 안전장치**: 클릭한 4점을 노트북의 `order_quad`와 동일하게(중심 기준 각도 정렬 →
  좌상단부터 시작 → 외적 부호로 시계방향 보정) 다시 정렬해서, 클릭 순서가 살짝 어긋나거나
  사진이 회전되어 있어도 사각형이 꼬이지 않게 함.
- **출력 크기**: 노트북의 `BOARD_OUTPUT_SIZE`와 동일하게 기본 850×550px 고정 캔버스로
  원근 변환 (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`, `INTER_CUBIC`). UI에서
  가로/세로 값을 바꿀 수 있음.
- 평탄화 이후 단계(추가 자르기 → 확대 → 탐지 → 검수 → 저장)는 다른 페이지들과 완전히
  동일하며, 결과는 같은 `RUMMIKUB_DATA_ROOT`(`rummikub_data_2`)에 저장됩니다.

### 사용 방법 (예: `evaluate_ui_batch.py`로 폴더 일괄 라벨링)

1. `.\abcd\Scripts\python.exe evaluate_ui_batch.py` 실행 → http://127.0.0.1:7862 접속
2. 라벨링할 새 이미지가 있는 폴더 경로를 입력하고 "폴더 불러오기"
3. (선택) 자르기·확대 → "탐지 실행"으로 `best.pt` 초안 생성
4. 표에서 중복/오탐 행 삭제, 클래스 오류 셀 더블클릭으로 수정, 놓친 타일은
   색상/숫자를 고른 뒤 결과 이미지를 두 번 클릭(두 모서리)해서 박스 추가
5. `split`(train/val/test) 선택 후 "검수한 결과 데이터셋으로 저장" →
   `C:\Users\SSAFY\workspace\rummikub_data_2\images\<split>\`,
   `...\labels\<split>\`에 이미지(.jpg)와 YOLO 라벨(.txt)이 저장됨
6. 저장하면 자동으로 다음 이미지로 넘어감 (같은 이미지를 다르게 잘라 여러 샘플로
   저장하고 싶으면 "다음 이미지"를 누르기 전까지 반복 가능)

## 5. 요약 체크리스트

새로 이 프로젝트를 clone/복사해서 시작할 때 순서대로 진행하면 됩니다.

1. `.\setup.ps1` 실행 → `abcd` 가상환경 생성 및 의존성 설치, `boardgame-ai` 커널 등록
2. 루트에 `.env` 생성 후 `ROBOFLOW_API_KEY` 입력 (템플릿 없으므로 직접 작성,
   또는 `legacy_roboflow/.env.example` 형식을 참고)
3. (레거시 도구를 쓸 경우) `legacy_roboflow\.env.example` → `legacy_roboflow\.env` 복사 후 키 입력
4. 루트에 `best.pt` 복사 (이미 완료됨). `dataset.yaml`/`rummikub/classes.py`는 이미
   `best.pt`의 클래스 순서에 맞춰뒀으므로 추가 작업은 필요 없음
5. `.\abcd\Scripts\python.exe import_roboflow.py` 실행 → `data/images`,
   `data/labels` 채우기 (공개 라벨, 검수 완료된 정답)
6. (선택) 직접 촬영한 이미지가 있다면
   `.\abcd\Scripts\python.exe auto_label.py --source <이미지폴더> --split train` 로
   `best.pt`가 초안 라벨을 일괄 생성 → `studio.py`의 "기존 라벨 검수"에서 사람이 확인
7. `.\abcd\Scripts\python.exe validate_dataset.py` 로 데이터 검증 (라벨 형식 +
   `dataset.yaml` 클래스 순서 일치 여부 확인)
8. `.\abcd\Scripts\python.exe evaluate.py --split test` 로 `best.pt`의 정밀도/재현율/mAP 확인
   (검수된 정답 라벨 기준이어야 의미 있음)
9. `.\abcd\Scripts\python.exe studio.py` 로 추가 라벨링 / 턴 검증 UI 사용
10. (선택) `.\abcd\Scripts\python.exe train.py --epochs 100 --imgsz 960` 로 `best.pt`를
    추가 파인튜닝 → `runs/detect/rummikub_tiles/weights/best.pt` 생성
