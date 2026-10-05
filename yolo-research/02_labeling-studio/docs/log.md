# 작업 로그 (2026-07-23)

`yolo_boardgame`과 `rummikub-webcam` 두 프로젝트에 걸쳐 진행한 작업 정리입니다.

## 1. `yolo_boardgame` 파일 구성 문서화

- `docs/readme.md` 신규 작성: 전체 파일/디렉터리 용도, `.gitignore`로 제외되는 항목 중
  직접 준비해야 하는 것(`.env`, `.venv`, `data/images`·`data/labels` 등) 정리, 요약 체크리스트 포함.

## 2. `rummikub-webcam`: 85x110 테이블 crop + 평면화 노트북

- `test4.ipynb`(원본 vs 평면화 YOLO 탐지 비교)와 `holography.ipynb`(A4 평면화 로직)를 참고해
  `test5.ipynb` 작성 → 이후 사용자가 `test7.ipynb`로 정리.
  - `homography_table/`의 초록 테이프 4모서리를 검출해 테이블 실측 비율로 평면화, 결과를
    `homography_table_result/`에 저장.
  - 처음 110x170으로 만들었다가 85x110으로 수정.
  - `best_v8.pt`로 원본/평면화 탐지 결과 비교 섹션 포함.

## 3. `rummikub-webcam`: 배경색 비교 + Crop/평면화 (`test8.ipynb`)

- 45x35.5cm 테이프 기준, `background/`(흰색 책상 vs 검정 매트 두 배경) 대상 신규 노트북.
  1. 테이프 bounding box로 원본 crop (`_crop.jpg`)
  2. crop 이미지를 45:35.5 비율로 평면화 (`_flat.jpg`)
  3. 두 결과를 `background_result/`에 저장
  4. crop 이미지의 HSV 명도로 배경색(밝음/어두움) 자동 분류
  5. `best_v8.pt`로 원본/Crop/평면화 x 배경색 조합별 탐지 성능 비교 + 그룹 막대그래프
- 이어서 "평가 방법"에 대한 논의: 단순 탐지 개수 비교는 오탐/정탐을 구분 못하므로, 고정된
  12개 타일(정답)을 기준으로 정밀도/재현율을 계산하는 방식을 제안 (아직 노트북에 반영은 안 함).

## 4. `yolo_boardgame`: 모델 평가 기능 추가 (`evaluate.py`)

- `validate_dataset.py`는 라벨 형식만 검사하고 모델 성능은 측정하지 않는다는 점을 확인.
- `evaluate.py` 신규 작성: `dataset.yaml` split(train/val/test)에 대해 Ultralytics 평가 실행,
  정밀도/재현율/mAP50/mAP50-95 + 클래스별 mAP50-95 출력.
- `README.md`/`docs/readme.md`에 평가 섹션 및 체크리스트 추가.

## 5. `yolo26` 학습 → 기존 `best.pt` 활용으로 전환

- `rummikub-webcam/best.pt`를 `yolo_boardgame/best.pt`로 복사.
- `train.py`: `--model` 기본값을 `yolo26n.pt`(COCO 사전학습) → 루트 `best.pt`로 변경
  (새로 학습이 아니라 `best.pt`부터 파인튜닝하는 구조로 전환). `--name` 기본값도
  `rummikub_yolo26n` → `rummikub_tiles`로 정리.
- `evaluate.py`: `--weights` 기본값을 루트 `best.pt`로 변경.
- `studio.py`: `DEFAULT_WEIGHTS`, 자동 라벨링 UI 문구, "학습 안내" 탭 문구를 `best.pt` 기준으로 갱신.
- `README.md`/`docs/readme.md`: 소개 문구, 파일 표, 체크리스트를 `best.pt` 중심으로 재작성.

## 6. 클래스 순서 불일치 발견 및 해결

- `best.pt` 내부 바이트를 직접 조사해 클래스 순서가 `Black1, Black10, Black11, ..., Black9,
  Blue1, ..., Joker, Orange1, ..., Red9` (Roboflow 원본 라벨 텍스트의 **알파벳순**)임을 확인.
  `best_v8.pt`, `best_v26.pt`도 동일한 순서라 모델 버전을 바꿔도 해결되지 않는다는 것도 확인.
- `yolo_boardgame`의 `dataset.yaml`/`rummikub.classes.CLASS_NAMES`는 숫자순(`black_1`,
  `black_2`, ...)이었어서, 그대로면 `evaluate.py`의 클래스별 정밀도/재현율/mAP가 잘못 매칭됨.
- 해결(옵션 b, 라벨 재매핑 방식 선택):
  - `rummikub/classes.py`: `CLASS_NAMES`를 raw label(`Black1`, `Black10`, ...) 기준
    알파벳순으로 정렬해서 생성하도록 변경 (표시 이름은 `black_1` 스타일 유지, 순서만 변경).
  - `dataset.yaml`: `names:` 매핑을 동일한 순서로 재작성.
  - `validate_dataset.py`: `dataset.yaml`과 `CLASS_NAMES` 순서가 어긋나면 잡아내는
    `check_dataset_yaml()` 추가.
  - `evaluate.py`: `canonicalize_external_name`으로 정규화해 실제 순서 불일치일 때만
    경고하도록 수정.
  - 코드 전수 검색으로 `class_id // 13` 같은 순서 의존 로직이 없음을 확인 (전부
    `CLASS_TO_ID`/`CLASS_NAMES` 동적 조회라 재정렬에 안전). `data/labels`가 비어있어 기존
    라벨 마이그레이션은 불필요했음.

## 7. 두 프로젝트 가상환경 통합 시도 (`abcd`, Python 3.10)

- `rummikub-webcam/requirements.txt`(pip freeze 덤프)와 `yolo_boardgame/requirements.txt`
  비교 → `opencv-python`(4.13 vs 5.0, 메이저 버전 충돌), `numpy` 버전 차이, `torch`가
  `yolo_boardgame`엔 없음(CUDA wheel 별도 설치) 등 확인.
- `rummikub-webcam` 노트북들의 커널명이 이미 `boardgame-ai`(Python 3.10)로 설정돼 있었지만
  실제로 등록된 적은 없었음을 발견.
- 사용자가 Python 3.10 공유 가상환경 생성을 요청 → 위치는 `yolo_boardgame\.venv` 로 결정.
- 이후 사용자가 venv 이름을 `abcd`로 지정, 직접 생성까지 진행 요청:
  - `uv venv --python 3.10 abcd` (시스템에 Python 설치 없이 uv가 3.10 자동 다운로드)
  - `uv venv`가 만든 환경엔 `pip`이 없어서 `python -m pip install`이 실패 → `uv pip install
    --python abcd\Scripts\python.exe` 방식으로 전환
  - torch/torchvision(CUDA 13.0) 설치 성공
  - `requirements.txt` 설치 중 `numpy==2.4.4`가 Python **3.11+** 요구라 3.10과 충돌 →
    `numpy==2.2.6`으로 낮춰서 해결 (torch 설치 시 이미 깔린 버전과 동일)
  - `ipykernel install --name boardgame-ai`로 Jupyter 커널 등록 완료
  - 최종 검증: `torch.cuda.is_available() == True` (RTX 4070 Laptop GPU 인식), `cv2`/`gradio`
    등 정상 임포트, `best.pt` 로드 및 `validate_dataset.py` 통과 확인
- `setup.ps1`을 `py -3.10 -m venv` → `uv venv`/`uv pip install` 기반으로 재작성, `.venv` 경로를
  쓰던 `README.md`/`docs/readme.md`/`studio.py`/`.gitignore`/`rummikub-webcam/requirements.txt`
  전부 `abcd`로 갱신.

## 8. `rummikub-webcam`은 별도 가상환경 사용으로 재조정

- VS Code "Python: Select Interpreter" 목록에 `abcd`가 보이지 않는 문제 발생.
- `rummikub-webcam`은 공유 `abcd` 환경 대신 사용자가 직접 고르는 별도 가상환경을 쓰기로 함.
- `rummikub-webcam/requirements.txt`를 다시 독립 설치 가능한 파일로 작성:
  - 실제 이 프로젝트가 import하는 패키지만 포함: `opencv-python`, `numpy`, `matplotlib`,
    `torch`, `torchvision`, `ultralytics`, `ipykernel` (`PyYAML`/`requests`는 쓰이지
    않아서 제외).
  - 버전을 `==` 정확 고정 대신 `>=` 최소 버전으로 바꿔서, 어떤 Python 버전의 가상환경을
    선택하든 pip가 알아서 호환 버전을 resolve하도록 함.
  - CUDA 빌드가 필요하면 별도 `--index-url` 명령을 쓰라는 안내 주석 추가.
- `yolo_boardgame`의 `abcd` 환경/`setup.ps1`은 그대로 유지 (yolo_boardgame 자체는 계속
  이 환경을 씀).

## 9. 기존 `best.pt`로 폴더 단위 자동 라벨링하는 기능 추가 (`auto_label.py`)

- "기존 가중치를 활용해서 라벨링하고 결과를 평가하는 기능"이 실제로는 구현 안 돼 있었음을
  확인 — `data/images`·`data/labels`가 비어 있어 `evaluate.py`를 돌려도 평가할 데이터가
  없었고, `studio.py`의 자동 라벨링은 이미지 한 장씩 UI로 돌리는 방식이라 폴더 단위 배치
  처리가 불가능했음.
- `auto_label.py` 신규 작성: `--source` 폴더의 이미지를 `best.pt`로 일괄 추론해
  `data/images/{split}`, `data/labels/{split}`에 YOLO 라벨로 저장.
  - `rummikub/smart_label.py`의 `local_yolo_auto_boxes`(이미 `canonicalize_external_name`으로
    `best.pt`의 raw 클래스명 `Black1` 등을 `black_1`로 정규화하는 로직 포함)와
    `rummikub/annotations.py`의 `save_yolo_sample`을 그대로 재사용해 `studio.py`의 대화형
    자동 라벨링과 완전히 같은 경로를 탐.
  - `--confidence`, `--allow-empty`(무검출 이미지도 저장할지) 옵션 제공.
  - import/CLI(`--help`) 동작은 `abcd` 환경에서 확인. 실제 `data/images`에 쓰는 실행은
    사용자가 원하는 이미지 폴더로 직접 돌리도록 남겨둠(임의 폴더로 테스트 실행하면 실제
    데이터셋에 원치 않는 데이터가 섞일 수 있어서 보류).
- ⚠️ 중요: 이 스크립트가 만드는 라벨은 모델 자신의 예측이라 **정답이 아님**. `studio.py`의
  "기존 라벨 검수"에서 사람이 확인/수정한 뒤에만 `evaluate.py`의 신뢰할 수 있는 정답으로
  쓸 수 있음 (검수 없이 바로 평가하면 모델이 자기 예측과 비교되어 의미 없이 높은 점수만 나옴).
- `README.md`(사용법 섹션), `docs/readme.md`(파일 표, 체크리스트 6번 항목)에 반영.

## 10. 이미지 1장 검수 UI (`evaluate_ui.py`) 신규 제작 및 반복 개선

- 처음엔 이미지 업로드 → `best.pt` 탐지 → 결과 이미지 + 탐지 목록만 보여주는 단순한
  Gradio 페이지(`evaluate_ui.py`, 포트 7861)로 시작.
  - 만드는 과정에서 `rummikub/detector.py`의 `detect_tiles`가 `best.pt`의 raw 클래스명
    (`Black1` 등)을 정규화 없이 그대로 반환한다는 걸 발견 — `canonicalize_external_name`으로
    `black_1` 형식으로 바꿔서 표시하도록 수정.
- 실행 중 `abcd` 가상환경 폴더가 원인 불명으로 통째로 사라지는 문제 발생 → `uv venv`부터
  torch/requirements/`ipykernel` 등록까지 처음부터 다시 진행해서 복구 (백신이 낯선 이름의
  venv 폴더를 격리했을 가능성 언급).
- 정답 텍스트를 입력해 정밀도/재현율을 계산하는 "채점" 기능을 1차로 추가했다가, 사용자
  요청으로 **박스 단위 검수 방식으로 교체**:
  - 탐지 결과를 클래스+신뢰도 요약이 아니라 **박스 1개당 1행**(`#, 클래스, 신뢰도,
    x1,y1,x2,y2`)인 편집 가능한 `gr.Dataframe`으로 표시.
  - IoU를 계산해 겹치는(≥0.6) 박스 쌍을 자동으로 찾아 요약란에 경고 — 사용자가 표에서
    직접 해당 행을 삭제(중복 제거)하거나 클래스 셀을 고칠 수 있음.
  - "데이터셋으로 저장" 버튼: 검수(삭제/수정)된 표 내용을 `rummikub/annotations.py`의
    `save_yolo_sample`로 새 위치 `C:\Users\SSAFY\workspace\rummikub_data`(별도 폴더,
    `yolo_boardgame/data`와 분리)에 YOLO 형식(이미지 + `.txt`)으로 저장. split(train/val/test)
    선택 가능.
- IoU 계산, 중복 탐지, 저장 로직을 합성 이미지로 직접 실행해 검증 (클래스 id가
  `CLASS_NAMES`/`best.pt` 순서와 일치하는 것까지 라벨 파일 내용으로 확인), 테스트 산출물은
  삭제하고 빈 폴더 구조만 남김.
- `README.md`(기능 목록, "이미지 1장 검수 UI" 섹션), `docs/readme.md`(파일 표)에 반영.

## 11. `evaluate_ui.py`: 클래스 클릭 시 해당 박스만 표시하는 필터 추가

- 표에서 행을 클릭하면(`gr.Dataframe.select`) 그 행의 클래스와 같은 박스만
  `rummikub/annotations.py`의 `draw_boxes`로 원본 이미지에 다시 그려서 보여줌 (같은
  클래스가 여러 개 있으면 전부 표시). "전체 보기" 버튼으로 다시 전체 박스로 복귀.
- `_row_to_box` 헬퍼로 표 행 → box dict 변환 로직을 `save_dataset`과 공유하도록 정리.
- 가짜 `SelectData`(`evt.index`)를 만들어 오프라인으로 필터링/전체보기/범위 밖 인덱스/잘못된
  클래스 케이스를 전부 검증.
- `README.md` 사용법에 3번 단계로 반영.

## 12. 클릭 시 오류 토스트 버그 수정 + 클래스 필터를 행 단위(1개)로 변경

- 배포 후 사용자가 행을 클릭하면 "오류" 토스트만 뜨는 버그 리포트 → 원인 조사.
  `abcd\Lib\site-packages\gradio\components\dataframe.py` 소스를 직접 열어 확인한 결과,
  `gr.Dataframe`의 `type` 파라미터 기본값이 `"pandas"`라서 콜백에 `list[list]`가 아니라
  `pandas.DataFrame`이 전달되고 있었음. `rows[row_index][1]` 같은 리스트 인덱싱이 컬럼
  탐색으로 해석되어 `KeyError` → 처리 안 된 예외 → Gradio 기본 오류 토스트로 이어졌던 것.
  `table_output`에 `type="array"`를 추가해 해결 (해당 컴포넌트를 입력으로 쓰는
  `filter_by_selected_class`/`show_all_boxes`/`save_dataset` 전부 영향받고 있었음).
- 이어서 "같은 클래스면 박스가 동시에 여러 개 뜬다"는 요청 → `filter_by_selected_class`를
  클래스 전체 매칭에서 **클릭한 행 1개만** 표시하도록 변경 (같은 클래스가 여러 개 있어도
  클릭한 박스 하나만 남김). 가짜 클릭 이벤트로 같은 클래스의 서로 다른 행을 클릭했을 때
  각각 자기 자신의 박스만 나오는지 확인.
- `README.md` 안내 문구를 "그 클래스 전부" → "그 박스 하나만"으로 수정.

## 13. 행 삭제 불가 / 열 삭제만 가능한 버그 수정 + 저장 포맷 설명

- "열 삭제만 되고 틀린 라벨(행)은 못 지운다"는 리포트 → `gradio/components/dataframe.py`를
  다시 직접 읽어 원인 확정. 이 Gradio 버전(6.20.0)부터 `row_count`/`col_count`의 튜플 인자가
  deprecated고, 실제로는 `row_count: int`(항상 dynamic으로 처리됨) + `column_count: int`
  (역시 항상 dynamic) + (아직 미구현인) `row_limits`/`column_limits`로 대체됐음.
  - 기존 코드가 `row_count=("dynamic", 0)`로 **순서가 반대**로 들어가 있어서, 내부적으로
    `("dynamic", 0)`가 그대로 저장되고 모드 문자열 자리에 정수 `0`이 들어가 프론트엔드가
    "dynamic"으로 인식하지 못해 행 삭제가 막혀 있었음.
  - `col_count=(7, "fixed")`는 **deprecated 파라미터라 실제 컬럼 개수 계산에 반영되지 않고**
    경고만 내는 용도였고, 실제 `column_count`는 한 번도 설정 안 돼서 기본값(동적)으로
    남아 컬럼이 삭제 가능한 상태였음 — 사용자가 겪은 증상과 정확히 일치.
  - 수정: `row_count=0`(일반 int, 자동으로 dynamic 처리) + `column_count=(7, "fixed")`
    (deprecated 체크가 없는 새 파라미터명에 튜플을 넣어 구조 고정, 셀 값 편집은 그대로 가능).
    `static_columns`는 changelog상 "셀 값 삭제까지 막는" 옵션이라 (틀린 클래스 수정 기능이
    깨질 수 있어) 채택하지 않음.
  - `gr.Dataframe(...)` 객체를 직접 생성해 `df.row_count == (0, "dynamic")`,
    `df.column_count == (7, "fixed")`, 경고 없음을 확인 후 반영.
- 저장 `.txt` 형식 질문에 답변: Ultralytics YOLO Detect 포맷 그대로
  (`<class_id> <center_x> <center_y> <width> <height>`, 좌표는 이미지 크기 대비 0~1
  정규화, 소수점 6자리, 박스 1개당 한 줄). `class_id`는 `rummikub.classes.CLASS_NAMES`
  인덱스(= `best.pt` 학습 순서, 숫자순 아님) 기준.

## 14. 자르기 → 확대 → 탐지 파이프라인 추가, 저장 대상을 "자른 사진"으로 변경

- 요청: 분석 전에 자르기 가능, 자른 다음 확대, 그다음 분석 진행. 저장은 자른 사진 기준으로.
- 기존 `gr.Image` 단일 업로드 입력을 `gr.ImageEditor`(`transforms=("crop",)`, `image_mode="RGB"`,
  `eraser=False`, `brush=False`, `layers=False`로 자르기 도구만 남김)로 교체 — Gradio
  내장 크롭 도구 사용, 소스 코드(`image_editor.py`)로 `transforms` 옵션과 `type="numpy"`일 때
  콜백에 `{"background","layers","composite"}` dict가 전달되는 것 확인.
- 파이프라인을 상태 2개로 구성: `cropped_state`(자르기 결과, 배율 1x 고정) →
  `working_state`(확대 배율 적용 결과, 배율 슬라이더 바뀔 때마다 `cropped_state`에서 새로
  계산 — 압축 열화 누적 방지). 탐지/검수/저장 전부 `working_state`를 사용하도록 배선해서
  "화면에 보이는 것 = 분석한 것 = 저장되는 것"이 항상 일치하도록 함 (좌표계 불일치 방지).
  확대를 안 하면(기본 1.0배) `working_state`는 자른 이미지와 동일.
- `apply_crop`(composite 없으면 background로 폴백, 빈 값 처리), `apply_zoom`(1.0배면 원본
  그대로 반환해 불필요한 재인코딩 방지) 둘 다 합성 이미지로 오프라인 검증.
  `gr.ImageEditor`/`build_app()` 인스턴스 생성도 별도로 검증(경고 없음).
- ⚠️ 브라우저에서 크롭 도구를 실제로 드래그하는 상호작용 자체는 (브라우저 자동화 도구가
  없어) 직접 확인하지 못함 — Gradio 문서/소스코드 기준으로 구현. 크롭 UI가 기대와 다르게
  동작하면 알려달라고 안내함.
- `README.md` "이미지 1장 검수 UI" 섹션을 1~5단계(자르기/확대/탐지/검수/저장)로 재작성,
  저장이 "자른 이미지(확대 적용 시 그 상태) 기준"임을 명시.

## 15. `gr.ImageEditor` 크롭 도구가 안 보임 → 슬라이더 기반 자르기로 교체

- 사용자 리포트: "계속 업로드만 되는데 자르기 어떻게 해?" — `gr.ImageEditor`의 크롭
  도구가 화면에서 발견/동작이 안 됨. 브라우저 상호작용을 직접 볼 방법이 없어 원인을
  확정할 수 없었음(우려했던 대로 14번 항목에서 남긴 경고가 실제로 발생).
- 신뢰도를 우선해 `gr.ImageEditor` 의존을 완전히 제거하고, 기본 컴포넌트(`gr.Image` +
  `gr.Slider` 4개 + `cv2.rectangle` 미리보기)만으로 자르기를 재구현:
  - 이미지를 올리면(`raw_image_input.change`) 슬라이더 4개(x1,y1,x2,y2)의 최소/최대값을
    이미지 크기에 맞춰 자동 세팅(`on_image_uploaded`).
  - 슬라이더를 놓을 때마다(`.release()`) 원본 이미지 위에 빨간 사각형으로 자를 영역을
    미리 그려서 보여줌(`preview_crop_region`), 실제 자르기 전 확인 가능.
  - "자르기 적용" 클릭 시에만 실제로 numpy 슬라이싱으로 잘라서
    `cropped_state`/`working_state`에 반영(`apply_crop`) — 순서 뒤바뀐 좌표 자동 정렬,
    이미지 경계로 clamp, 너무 작은 영역(2px 미만) 방지까지 처리.
  - 확대(zoom) 이후 파이프라인(2~5단계)은 그대로 유지.
- 합성 이미지로 슬라이더 업데이트 값, clamp(정상/좌우반전/범위초과), 미리보기 shape,
  실제 crop 결과 shape, 너무 작은 영역 에러 메시지, 이미지 없음 케이스까지 전부
  오프라인 검증 후 반영.
- `README.md` 1단계 설명을 슬라이더 방식으로 갱신.

## 16. 슬라이더 대신 클릭(드래그 대체)으로 자르기 + 놓친 타일 박스 직접 추가

- 요청: (a) 모델이 못 찾은 타일은 마우스로 드래그해서 박스를 만들 수 있게, (b) 자르기와
  좌표 입력도 슬라이더/숫자 대신 마우스로 위치·크기를 조절하게 해달라는 요청.
- Gradio 순정 컴포넌트로는 실시간 드래그-리사이즈 핸들을 안전하게 구현할 방법이
  마땅치 않고(지난번 `ImageEditor` 신뢰성 문제 재발 우려), `studio.py`에 이미 검증된
  "두 번 클릭으로 박스 정의"(`add_corner`/`handle_canvas_click`) 패턴을 그대로 재사용하는
  쪽으로 결정 — 같은 환경에서 실제로 동작이 증명된 방식이라 신뢰도가 더 높음.
- 자르기: 4개 슬라이더를 없애고, `raw_image_input`(업로드 전용) →
  `raw_original_state`(클릭 좌표 계산용 원본 사본) → `crop_click_preview`(클릭 대상)
  구조로 변경. 첫 클릭은 `crop_pending_state`에 저장하고 노란 십자 마커만 표시,
  두 번째 클릭에서 사각형을 확정해 `crop_box_state`에 저장 → "자르기 적용" 버튼이
  그 상태를 읽어 실제로 자름. "모서리 선택 취소" 버튼으로 첫 클릭만 취소 가능.
  매번 `raw_original_state`(가공 안 된 원본)에서 다시 그리므로 마커가 누적되지 않음
  (studio.py와 동일한 설계).
- 박스 추가: 탐지 결과 이미지(`annotated_output`)를 `interactive=True`로 바꾸고(클릭 이벤트
  신뢰성을 위해 studio.py의 `preview` 설정과 동일하게 맞춤) 같은 두 번 클릭 패턴을
  적용(`handle_box_click`). 클래스는 새 드롭다운(`new_box_class_input`, `CLASS_NAMES` 사용)
  에서 먼저 고르고, 두 번째 클릭이 끝나면 `table_output`에 새 행이 자동 추가됨(신뢰도는
  사람이 만든 박스이므로 1.0으로 기록). "박스 추가 취소"로 첫 클릭 취소 가능.
- `rummikub/annotations.py`의 `draw_boxes(image, boxes, pending)`를 그대로 재사용해서
  박스 + 대기 중인 마커를 함께 그림 — studio.py가 쓰는 것과 동일한 함수라 동작 방식이
  이미 검증돼 있음.
- 가짜 클릭 이벤트로 자르기 2클릭 완주, 너무 작은 영역 취소, 대기 취소, 박스 추가
  2클릭 완주(다음 인덱스 번호 자동 계산 포함), 박스 추가 취소까지 전부 오프라인
  end-to-end 검증 후 반영.
- `README.md` 1·4단계를 클릭 기반 설명으로 재작성.

## 17. 자르기를 선택 사항으로 변경

- 지금까지는 `working_state`가 "자르기 적용"을 눌러야만 채워져서, 자르지 않으면
  탐지("먼저 이미지를 자르기까지 적용하세요" 에러)를 실행할 수 없었음.
- `on_raw_image_change`가 이미지를 업로드하는 즉시 `cropped_state`/`working_state`를
  원본 그대로로도 채우도록 변경 — 자르기를 안 해도 바로 2번(확대) or 3번(탐지)으로
  진행 가능. 자르기를 하면 그 결과가 그대로 두 state를 덮어써서 기존 동작은 동일하게
  유지됨.
- `evaluate_image`의 안내 문구를 "먼저 1번에서 이미지를 업로드하세요"로 단순화.
- 합성 이미지로 업로드 직후 `cropped_state`/`working_state`가 채워지는 것과, 이미지
  없음(`None`) 케이스 둘 다 오프라인 검증.
- `README.md` 1단계를 "자르기 (선택)"으로 갱신.

## 18. 영역 미선택 상태로 "자르기 적용" 누르면 진행이 막히던 버그 수정

- 리포트: "선택 안 하니까 자르기 적용이 반응 안 해서 다음으로 안 넘어가는데?" — 17번에서
  자르기를 선택 사항으로 만들었지만, `apply_crop`은 `crop_box`가 `None`(영역 미선택)이면
  여전히 에러 메시지와 함께 `cropped_state`/`working_state`에 `None`을 반환하고 있었음.
  즉 업로드 직후 자동으로 채워둔 작업 이미지를 "자르기 적용" 클릭이 다시 지워버리는
  회귀였음.
- 수정: 영역 미선택 상태에서 "자르기 적용"을 누르면 에러 대신 **원본 이미지 전체를
  그대로 사용**하도록 변경(사실상 no-op) — 선택 여부와 상관없이 버튼을 누르면 항상
  유효한 작업 이미지가 유지되도록 함.
- 합성 이미지로 "영역 미선택 + 적용", "이미지 자체가 없음" 두 케이스 오프라인 검증.

## 19. 절반 이상 겹치는 박스를 자동으로 삭제하도록 변경 (경고만 하던 것 → 실제 제거)

- 기존 `find_duplicate_pairs`는 IoU ≥ 0.6인 박스 쌍을 찾아 "표에서 지우세요"라고
  경고만 하고 실제로 지우지는 않았음. 요청은 절반 이상(IoU > 0.5) 겹치면 자동으로
  지워달라는 것.
- `find_duplicate_pairs`를 `remove_overlapping_duplicates`로 교체: 신뢰도 내림차순으로
  훑으면서 이미 채택한 박스와 IoU > `DUPLICATE_IOU_THRESHOLD`(0.6→**0.5**로 조정)인
  후보는 버리는 그리디 NMS. **클래스가 달라도** 겹치면 중복으로 처리하는데, 같은
  물리적 타일을 모델이 다른 클래스로 헷갈려 중복 예측하는 경우 Ultralytics 자체 NMS는
  클래스별로만 동작해서 못 걸러내기 때문. 최종 결과는 원래 탐지 순서를 유지하고
  "#" 번호를 1부터 다시 매김.
- `evaluate_image`가 탐지 직후 이 함수를 호출해 표를 정리하고, 결과 이미지도 정리된
  박스 기준으로 다시 그리도록 변경(`draw_boxes`로 재렌더링). 요약란에 "N개 → 중복 제거
  후 M개"와 자동 삭제 개수를 표시.
- 합성 박스 세트(다른 클래스지만 겹치는 쌍 2개, 안 겹치는 박스 1개, 빈 리스트)로
  제거 개수·순서 유지·번호 재부여까지 오프라인 검증.
- `README.md` 3단계 설명을 "자동 삭제"로 갱신.

## 20. 박스 추가 시 결과 이미지와 표가 따로 노는 문제 → 표를 단일 진실 공급원으로 변경

- 리포트: "박스를 추가하면 이미지에는 나오는데 목록에는 안 나오고 있어" — `handle_box_click`
  자체는 두 클릭이 끝나면 `rows`에 새 행을 append해서 `table_output`으로 반환하고 있어
  로직상으로는 맞았지만(오프라인 테스트에서도 확인됨), 브라우저에서 이미지와 표 두
  컴포넌트가 같은 함수 호출의 서로 다른 출력값이다 보니 반영이 어긋나는 걸로 보임.
- 근본 원인을 확정할 수 없어(브라우저 상호작용 직접 확인 불가), 대신 **구조적으로
  어긋날 수 없게** 재설계: `redraw_from_table(image, rows)`를 추가하고
  `table_output.change(...)`에 연결 — 표 데이터가 **어떤 이유로 바뀌든**(클릭으로 추가,
  행 삭제, 셀 직접 수정, 자동 탐지 등) 결과 이미지를 표 기준으로 항상 다시 그리도록 함.
  이제 표가 유일한 진실 공급원이고 이미지는 그 렌더링 결과일 뿐이라, 개별 핸들러의
  직접 반환값이 어떻게 되든 최종적으로는 항상 표와 일치하게 됨.
- 기존 핸들러들의 직접 `annotated_output` 반환은 그대로 유지(즉시 피드백용, 특히 첫
  클릭의 마커 표시처럼 표가 안 바뀌는 경우) — cascade와 중복 계산되는 경우가 있지만
  같은 결과를 다시 그리는 것뿐이라 무해함.
- 합성 이미지로 `redraw_from_table`이 박스를 정확히 그리는지, 이미지 없음 케이스, 전체
  앱 빌드까지 오프라인 검증.

## 21. 박스 추가용 클래스 선택을 색상/숫자 드롭다운 2개로 분리, 색상은 유지

- 기존 `new_box_class_input` 단일 드롭다운(`black_1`, `red_10`, `joker` 등 53개 값)을
  `new_box_color_input`(black/blue/orange/red/joker, `rummikub.classes.COLORS` 재사용)과
  `new_box_number_input`(1~13) 두 개로 분리. `combine_class_name(color, number)`가 둘을
  합쳐서 실제 클래스명(`joker`면 숫자는 무시)을 만듦 — 53가지 조합 전부 실제
  `CLASS_TO_ID`에 존재하는지 오프라인으로 전수 검증.
- `handle_box_click` 시그니처를 `class_name` 하나 대신 `color, number` 두 인자로 변경.
- "색상은 마지막 입력값 유지" 요구는 별도 구현 불필요 — Gradio는 어떤 핸들러의
  `outputs`에도 포함되지 않은 컴포넌트 값을 자동으로 유지하므로, 두 드롭다운을 어떤
  `outputs=[...]`에도 넣지 않는 것만으로 이미 요구사항을 만족함(숫자도 마찬가지로
  유지되며, 필요시 색만 유지하고 숫자만 초기화하는 것도 이후 요청 가능).
- 가짜 클릭 이벤트로 2클릭 완주 후 `combine_class_name` 결과가 표에 정확히 들어가는지,
  전체 앱 빌드까지 오프라인 검증.
- `README.md` 4단계 설명을 색상/숫자 분리 + 색상 유지 언급으로 갱신.

## 22. 폴더 일괄 검수 페이지 신규 (`evaluate_ui_batch.py`)

- 요청: 같은 기능이지만 폴더를 올리면 그 폴더의 모든 파일을 대상으로, 하나 저장되면
  다음 이미지로 자동으로 넘어가는 새 페이지.
- 로직을 중복 작성하지 않고 `evaluate_ui.py`에서 크롭/확대/탐지/검수/저장 관련 함수를
  전부 그대로 `import`해서 재사용(`on_raw_image_change`, `handle_crop_click`,
  `apply_crop`, `apply_zoom`, `evaluate_image`, `handle_box_click`, `filter_by_selected_class`,
  `show_all_boxes`, `redraw_from_table`, `save_dataset` 등) — 새로 작성한 코드는 폴더
  탐색/이동 로직뿐.
- 폴더 업로드 위젯 대신 **폴더 경로를 직접 입력**받는 방식 채택 — 로컬 전용 툴이라
  경로 입력이 더 간단하고, 이번 세션에서 겪은 Gradio 컴포넌트 신뢰성 문제
  (`ImageEditor` 등)를 반복하지 않기 위해 검증하기 쉬운 방식을 우선함.
  `auto_label.py --source <폴더>`와도 동일한 관례.
- `load_folder`(폴더 안 이미지 목록화 + 첫 장 로드), `advance_to_next`/
  `_find_next_readable`(읽기 실패한 파일은 자동으로 건너뛰고 다음 파일 탐색),
  `save_and_advance`(저장 성공 시에만 다음 이미지로 이동, 실패하면 같은 이미지에 머묾),
  `skip_current_image`(저장 없이 다음으로) 구현.
- `raw_image_input`을 업로드 컴포넌트가 아니라 `interactive=False` 표시 전용으로 바꾸고
  폴더 탐색 함수들이 값을 채워 넣음 — 값이 바뀌면 기존 `on_raw_image_change`가 그대로
  `.change()`로 반응해서 크롭 상태를 초기화(재사용 그대로 동작). 추가로 이미지가
  바뀔 때 이전 이미지의 탐지 결과(표/이미지/요약)도 지우는
  `clear_detection_outputs`를 별도 `.change()` 리스너로 연결.
- 합성 이미지 3장 + 손상된 파일 1개로 폴더 로드, 정상 진행, **손상 파일 자동 건너뛰기**,
  끝까지 도달, 수동 건너뛰기, 저장 실패 시 인덱스 유지, 잘못된 경로 처리까지 전부
  오프라인 검증 후 반영. 테스트에 쓴 이미지 파일은 삭제.
- 포트는 `evaluate_ui.py`(7861)와 동시에 켤 수 있도록 7862로 분리. 두 서버 모두 정상
  기동 확인.
- `README.md`에 "폴더 일괄 검수 UI" 섹션 추가, `docs/readme.md` 파일 표에 행 추가.

## 23. 청테이프(파란 테이프) 평탄화가 포함된 폴더 일괄 검수 페이지 신규 (`evaluate_ui_flatten.py`)

- 요청: 새 페이지에서 폴더를 입력받으면 청테이프 모서리로 가로/세로 길이를 입력받아
  평탄화한 뒤 데이터 라벨링까지 진행.
- `rummikub-webcam` 프로젝트의 `holography.ipynb`/`test5~8.ipynb`에서 쓰던 초록 테이프
  코너 검출 + `getPerspectiveTransform`/`warpPerspective` 평탄화 로직을 그대로 옮겨오되,
  색만 초록(H 35-85) 대신 파란 테이프(H 95-130 기본값, UI 슬라이더로 조정 가능)로 변경.
  `order_points`(각도 기준 회전 안정적 정렬), `detect_tape_corners`, `warp_to_ratio`
  (가로/세로 cm 중 어느 게 더 긴지 몰라도 `sorted()`로 자동 판별, 촬영 방향과 무관하게
  비율 매칭)를 이 파일에 재구현.
- BGR/RGB 색공간을 주의해서 처리: 테이프 검출은 반드시 `cv2.imread`가 준 원본 BGR
  배열로 수행(HSV 변환이 채널 순서에 의존하므로, RGB로 먼저 바꾸면 파란색이 빨간색
  Hue로 잘못 인식됨) — 파이프라인의 나머지(RGB 기대)로 넘기기 직전에만 변환.
- 폴더 탐색은 이미지 픽셀이 아니라 **현재 파일 경로**(`current_path_state`)만 들고
  있다가, `.then()` 체이닝으로 탐색 함수 다음에 `flatten_current`를 이어붙여 이미지가
  바뀔 때마다 자동으로 평탄화가 실행되도록 구성(폴더 불러오기/건너뛰기/저장 후 다음
  이미지 전부 동일하게 체이닝). 테이프 인식 실패 시 원본을 그대로 보여주고 경고만 하며
  중단시키지 않음 — Hue 슬라이더 조정 후 "평탄화 다시 시도"로 재시도하거나, 1-1번
  수동 자르기(기존 `evaluate_ui.py` 크롭 로직 재사용)로 대체 가능.
- 평탄화 이후 크롭/확대/탐지/검수/저장은 전부 `evaluate_ui.py`에서 직접 import해서
  재사용(중복 구현 없음).
- 합성 이미지(4개 파란 사각형)로 코너 검출 위치·정렬, 평탄화 후 종횡비가 입력한
  가로:세로와 일치하는지(허용오차 내), 테이프 없는 이미지의 폴백 동작, 실제 파일
  기반 `flatten_current`, 폴더 탐색 전체까지 오프라인 검증 후 테스트 파일 삭제.
- 포트는 7861/7862와 겹치지 않게 7863으로 분리. 서버 정상 기동 확인.
- `README.md`에 새 섹션 추가, `docs/readme.md` 파일 표에 행 추가.

## 24. 이미지 불러오기와 평탄화 단계를 분리 (자동 실행 → 버튼으로 명시적 실행)

- 요청: 지금은 이미지를 불러올 때 평탄화가 같이 진행되는데, 두 프로세스를 분리해달라.
- 기존엔 `load_folder`/`advance_to_next`(건너뛰기·저장 후 다음 이미지)에 `.then()`으로
  `flatten_current`를 체이닝해서 이미지가 바뀔 때마다 자동으로 평탄화가 실행됐음.
- 모든 `.then()` 체이닝 제거. 대신:
  - 탐색 함수들(`load_folder`, `advance_to_next`, `skip_current_image`,
    `save_and_advance`)은 이제 **원본 미리보기**(`raw_loaded_preview`, 평탄화 전
    RGB 이미지)만 채워 넣고, 검수 파이프라인 입력(`raw_image_input`)은 **항상 `None`으로
    비워서** 이전 이미지의 평탄화 결과가 남아있지 않게 함.
  - "평탄화 다시 시도" 버튼을 "**평탄화 실행**"으로 이름을 바꿔 유일한 트리거로 만듦 —
    `current_path_state`와 가로/세로/Hue 입력을 읽어 `flatten_current`를 실행하고
    `raw_image_input`을 채움(이후 크롭/확대/탐지 파이프라인은 그대로 이어짐).
- UI 순서도 "0. 폴더 불러오기"(경로+원본 미리보기만) / "1. 평탄화"(가로세로·Hue 입력+
  버튼, 여기서만 실행됨)로 재구성.
- 합성 이미지로 `load_folder`/`advance_to_next`/`skip_current_image`/`save_and_advance`가
  전부 평탄화 결과 슬롯에 `None`만 반환하는지(자동 평탄화 없음), 반대로
  `flatten_current`를 직접 호출했을 때만 실제로 평탄화되는지 오프라인 검증.
- `README.md`에 "불러오기와 평탄화는 분리된 단계"임을 명시.

## 25. 저장과 "다음 이미지"를 분리 (같은 이미지에서 여러 영역을 각각 저장 가능하게)

- 요청: 자르기 한 번 → 저장 → 다른 부분을 또 잘라서 작업할 수 있도록, 저장하면 자동으로
  다음 이미지로 넘어가는 대신 "다음" 버튼을 따로 눌러야 넘어가게 변경.
- `evaluate_ui_batch.py`, `evaluate_ui_flatten.py` 둘 다 동일하게 수정: `save_and_advance`/
  `skip_current_image`를 없애고, "저장"은 `save_dataset`을 그대로 호출만 하고(현재
  이미지에 머무름), "다음 이미지" 버튼(`go_to_next_image`)만 폴더 위치를 이동시키도록
  완전히 독립시킴. 저장을 여러 번 해도 같은 이미지에 계속 머무르므로, 크롭 영역을
  바꿔가며 여러 샘플을 뽑아낼 수 있음.
- 두 파일 다 합성 이미지로 "저장해도 인덱스 불변" + "다음 버튼을 눌러야만 인덱스 이동"을
  오프라인 검증.
- ⚠️ **중요 발견**: 정리 중 `rummikub_data/images(labels)/train`에 2026-07-23 15:53~17:46
  사이에 저장된 **실제 사진 크기의 파일 137장**이 이미 있는 것을 확인함 — 내 테스트용
  합성 이미지(수 KB짜리)와는 명백히 다른, 사용자가 그동안 페이지를 직접 사용하며 쌓은
  실제 데이터로 판단됨. 방금 만든 테스트 파일 2개만 정확히 식별해서 지우고 나머지
  137장은 전혀 건드리지 않음.
- 세 페이지(`evaluate_ui.py` 7861, `evaluate_ui_batch.py` 7862, `evaluate_ui_flatten.py`
  7863) 전부 재기동해서 모두 정상 응답 확인.

## 26. 밝기 조정 전용 페이지 신규 (`brightness_ui.py`, YOLO 무관)

- 요청: YOLO 기능 없이, 이미지 올리면 밝기만 조정하는 새 독립 페이지.
- `adjust_brightness(image, brightness)`: RGB에 값을 단순히 더하는 대신 HSV로 변환해
  **V(명도) 채널만** 가감(0~255로 clip) 후 다시 RGB로 변환 — 바로 전 대화에서 설명한
  "Hue는 그대로 두고 V만 조정" 원리를 그대로 적용해, RGB 각 채널에 동일하게 더할 때
  생기는 색조 왜곡을 피함.
  - `brightness == 0`이면 계산 없이 원본을 그대로 반환(무손실 no-op).
- 업로드/슬라이더 둘 다 `.change()`로 연결해 슬라이더를 움직이는 즉시 미리보기 갱신.
  별도 저장 버튼은 두지 않고 Gradio `gr.Image` 기본 다운로드 아이콘을 그대로 사용
  (YOLO 데이터셋과 무관한 범용 도구라 `rummikub_data` 같은 고정 저장 경로가 필요 없음).
- 합성 이미지로 밝게/어둡게 시 평균 V가 정확히 그만큼 변하는지, Hue는 거의 안 변하는지
  (색 왜곡 없음 확인), 흰색/검은색 극단에서 클리핑이 올바른지(오버플로/음수 없음),
  `brightness=0`이 진짜 원본과 동일한지(`np.array_equal`)까지 오프라인 검증.
- 포트 7864로 분리(7861~7863과 겹치지 않음). 서버 정상 기동 확인.
- `README.md`에 새 섹션 추가(겸사겸사 4-5단계 옛 "자동으로 다음 이미지" 문구가 25번
  변경 후 남아있던 것도 같이 수정), `docs/readme.md` 파일 표에 행 추가.

## 27. `rummikub_data` 밝기/좌우반전 증강 스크립트 작성 + 실행 (`augment_dataset.py`)

- 요청: 밝기 -25/0/+25 3종 × 좌우반전 유무 = 총 6종 증강 이미지를 추가하고, 기존
  라벨도 그에 맞게 변환.
- `brightness_ui.py`의 `adjust_brightness`(HSV V채널 조정)와 `evaluate_ui.py`의
  `RUMMIKUB_DATA_ROOT`/`SPLITS`를 그대로 import해서 재사용.
- 좌우반전 시 라벨은 `center_x -> 1 - center_x`만 미러링(class_id/center_y/width/height는
  불변), 밝기만 바뀌는 경우는 좌표 변화가 없으니 라벨을 그대로 복사.
- 재실행 안전장치: 생성 파일명이 `_bm25_orig`/`_b0_flip`/`_bp25_flip` 등 고정 접미사로
  끝나서, 나중에 원본이 더 늘어난 뒤 다시 돌려도 이미 증강된 파일을 또 증강하지 않음
  (`is_augmented()`로 스캔 단계에서 제외).
- 합성 데이터(박스 2개짜리, 빈 라벨의 negative sample, 라벨 없는 이미지)로 6종 생성 개수,
  좌우반전 좌표 수식, 빈 라벨 처리, 라벨 없음 스킵, 재실행 시 원본 카운트 불변(idempotent)
  까지 오프라인 검증.
  - 검증 중 "밝기 0 + 원본"이 저장 전 이미지와 완전히 똑같지 않다고 나온 이슈가 있었는데,
    실제 버그가 아니라 **JPEG를 다시 저장하면서 생기는 재압축 노이즈**(평균 오차
    0.08/255, 최대 3/255)였음을 별도 테스트로 확인 — 밝기 계산 자체(brightness=0은
    무손실 no-op)는 `brightness_ui.py` 테스트에서 이미 `np.array_equal`로 검증됨.
- 실행 전 실제 데이터 137장(→ 스크립트 실행 시점엔 149장으로 늘어나 있었음, 그 사이
  사용자가 페이지로 더 저장한 것으로 보임) 대상으로 대량 파일 생성(6배)이라 사용자
  확인을 받은 뒤 실행.
- 실제 실행 결과: `train` 원본 149장 → 증강 894장 저장(라벨 없음/읽기 실패 0장) →
  `images/train`, `labels/train` 각각 1043개로 정확히 일치(149+894). `val`/`test`는
  데이터가 없어 스킵.

## 28. `rummikub_data_2` 기존 라벨 검수/수정 페이지 신규 (`review_labels_ui.py`)

- 요청: `rummikub_data_2`의 이미지와 라벨을 겹쳐서 보여주고 검토/수정하는 페이지 추가.
  이전 페이지들(YOLO 모델로 새로 탐지 → 저장)과 달리, 이미 존재하는 라벨 파일을 그대로
  불러와 화면에 그린 뒤 고치는 용도라 모델 가중치가 필요 없음.
- `images/{split}/*.jpg`와 `labels/{split}/*.txt`를 쌍으로 순차 로드. YOLO 정규화 좌표
  (`class_id cx cy w h`)를 픽셀 `x1,y1,x2,y2`로 역변환해 표에 채우고, `CLASS_NAMES[class_id]`로
  클래스명을 붙여 `draw_boxes`로 오버레이.
- `evaluate_ui.py`의 박스 편집 도구(`handle_box_click`, `cancel_box_pending`,
  `filter_by_selected_class`, `show_all_boxes`, `redraw_from_table`, `_row_to_box`)를 그대로
  import해서 재사용 — 표에서 행 삭제/클래스 수정, 이미지 클릭 2회로 놓친 타일 추가 가능.
- 가장 큰 차이점: **저장 시 새 파일을 만들지 않고 원본 라벨 파일(`labels/{split}/{stem}.txt`)을
  덮어씀** (`normalize_box`로 픽셀 좌표를 다시 정규화). 새 샘플을 만드는 `save_yolo_sample`
  대신 직접 `Path.write_text`로 같은 경로에 저장.
- "불러오기"(폴더+split 지정 후 첫 이미지 로드)와 "다음 이미지"를 분리된 버튼으로 구현
  (앞선 페이지들의 로드/이동 분리 패턴과 동일).
- 합성 데이터(임시 폴더에 이미지 2장 + 라벨 텍스트)로 라벨 파싱 → 클래스명 변환 →
  행 삭제 후 저장(원본 파일 덮어쓰기 확인) → 다음 이미지 이동 → 마지막 이미지 이후
  처리까지 오프라인 검증 완료. 포트 7865로 분리해 서버 기동 후 정상 응답(HTTP 200)
  확인 후 종료.

## 현재 상태 요약

| 항목 | 상태 |
|---|---|
| `yolo_boardgame/abcd` | Python 3.10, 설치 및 CUDA 확인 완료. `boardgame-ai` Jupyter 커널로 등록됨. 한 번 원인 불명으로 삭제됐다가 복구됨 (백신 의심) |
| `yolo_boardgame/best.pt` | 복사 완료, 클래스 순서를 `dataset.yaml`/`CLASS_NAMES`에 맞춰 정렬 완료 |
| `yolo_boardgame/auto_label.py` | 작성 및 import/CLI 확인 완료. 실제 이미지로 실행은 아직 안 함 |
| `yolo_boardgame/evaluate_ui.py` | 클릭 기반 자르기 → 확대 → 탐지 → 박스 단위 검수(삭제/수정/클릭으로 놓친 타일 추가) → `rummikub_data`로 데이터셋 저장까지 구현, 로직 오프라인 검증 완료. 현재 서버 실행 중 (http://127.0.0.1:7861) |
| `yolo_boardgame/evaluate_ui_batch.py` | 폴더 경로 입력 → 이미지 순차 로드. 저장과 "다음 이미지"가 분리된 동작(같은 이미지에서 여러 번 저장 가능), 나머지는 `evaluate_ui.py`와 동일(함수 재사용). 로직 오프라인 검증 완료. 현재 서버 실행 중 (http://127.0.0.1:7862) |
| `yolo_boardgame/evaluate_ui_flatten.py` | 폴더 불러오기·평탄화·저장·다음 이미지가 전부 분리된 독립 버튼(자동 연쇄 없음) + `evaluate_ui.py` 검수/저장 재사용. 로직 오프라인 검증 완료. 현재 서버 실행 중 (http://127.0.0.1:7863) |
| `yolo_boardgame/brightness_ui.py` | YOLO 무관 독립 페이지, HSV V채널 기반 밝기 조정 + 실시간 미리보기. 로직 오프라인 검증 완료. 현재 서버 실행 중 (http://127.0.0.1:7864) |
| `yolo_boardgame/augment_dataset.py` | 밝기(-25/0/+25) x 좌우반전 증강 스크립트, 라벨 변환 포함. 오프라인 검증 후 실제 데이터에 실행 완료 |
| `C:\Users\SSAFY\workspace\rummikub_data` | **실제 데이터 있음** — `images/labels/train` 원본 149장 + `augment_dataset.py`로 생성한 증강 894장 = 총 1043장. `val`/`test`는 0장(비어 있음) |
| `yolo_boardgame/data/images`, `data/labels` | 아직 비어 있음 (라벨링 전) |
| `rummikub-webcam` | 자체 가상환경(사용자가 별도 선택) 사용, `requirements.txt`는 최소 버전 명시로 독립 설치 가능 |
| `test8.ipynb` 평가 방법 개선 | 논의만 하고 아직 코드 미반영 (정답 12개 타일 기준 정밀도/재현율 계산) |
| `yolo_boardgame/review_labels_ui.py` | `rummikub_data_2` 기존 이미지+라벨을 겹쳐 보여주고 검수/수정, 저장 시 원본 라벨 파일을 덮어씀. YOLO 모델 불필요. 로직 오프라인 검증 완료. 서버 기동 확인 후 종료(포트 7865) |
| `C:\Users\SSAFY\workspace\rummikub_data_2` | `test` split에 실제 라벨링 데이터 8쌍 존재(`train`/`val`은 비어 있음). `evaluate_ui.py`의 `RUMMIKUB_DATA_ROOT`가 이 폴더를 가리키도록 되어 있어 새 저장도 여기로 쌓임 |
