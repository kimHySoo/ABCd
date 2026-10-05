# Rummikub YOLO26 Studio

이미 루미큐브 타일로 학습된 `best.pt`를 기본 모델로 사용하는 로컬 프로젝트입니다.
CC BY 4.0 공개 라벨은 추가 파인튜닝용 데이터로만 사용하고, 기존 Roboflow 모델은
자동 라벨 초안을 만드는 선택 기능으로 분리했습니다.

주요 기능:

- 업로드 또는 웹캠 촬영 이미지 라벨링
- Roboflow Universe 공개 라벨 데이터 가져오기 및 검수
- Roboflow/로컬 best.pt 자동 라벨 초안 생성
- MobileSAM 기반 한 번 클릭 스마트 Bounding Box
- 53개 루미큐브 클래스의 YOLO Detect 형식 저장
- Train/Validation/Test 데이터 분리와 데이터 검증
- RTX GPU를 이용한 best.pt 추가 파인튜닝(선택) 및 mAP/정밀도/재현율 평가
- 이미지 1장 단위로 탐지 결과를 확인/교정하고 `rummikub_data`에 학습용 데이터로 저장하는
  간단 검수 UI (`evaluate_ui.py`)
- 이전 턴과 현재 턴 이미지의 타일 검출 결과 비교
- 그룹, 연속, 최초 등록 30점, 테이블 타일 유실 규칙 검증
- 위반 시 마지막 정상 턴 이미지 반환

기존 Roboflow 구현은 `legacy_roboflow/`에 보존되어 있으며 새 시스템에서는
사용하지 않습니다.

## 1. 환경 구성

현재 PC에는 PyTorch CUDA 13.0, RTX 4050 환경이 이미 구성되어 있습니다.
가상환경은 `abcd` 폴더에 Python 3.10으로 만들며, `rummikub-webcam` 프로젝트의
노트북과 환경을 공유합니다(아래 참고). [uv](https://docs.astral.sh/uv/)로 관리하므로
시스템에 Python을 따로 설치할 필요가 없습니다(uv가 3.10을 알아서 받아옵니다).
다시 구성할 때는 PowerShell에서 실행합니다.

```powershell
.\setup.ps1
```

`setup.ps1`은 `abcd` 가상환경 생성, 의존성 설치에 이어 이 가상환경을
`boardgame-ai`라는 이름의 Jupyter 커널로도 등록합니다 — `rummikub-webcam`의
노트북(`test4.ipynb`, `test5.ipynb`, `test7.ipynb`, `test8.ipynb`,
`holography.ipynb`)이 이 커널을 그대로 사용하도록 만들어져 있습니다.

## 2. 라벨링 및 턴 검증 UI

```powershell
.\abcd\Scripts\python.exe studio.py
```

브라우저에서 세 개 탭을 사용할 수 있습니다.

1. **데이터 라벨링**: 기존 라벨 검수, 자동 초안, 한 번 클릭 스마트 박스 및 수동 박스
2. **턴 룰 검증**: 기준 턴 저장 → 현재 턴 촬영 → 판정 및 정상 턴 확정
3. **학습 안내**: 데이터 검사 및 학습 명령 확인

라벨은 `data/images/{train,val,test}`와 `data/labels/{train,val,test}`에
YOLO 형식으로 저장됩니다. 타일이 전혀 없는 배경 이미지도 별도 체크 후
negative sample로 저장할 수 있습니다.

라벨링 도구:

- **스마트 박스**: 클래스를 고르고 타일을 한 번 클릭하면 MobileSAM이 경계를 생성
- **수동 박스**: 타일의 반대쪽 두 모서리를 차례로 클릭
- **클래스 변경**: 새 클래스를 고른 뒤 기존 박스 내부 클릭
- **박스 삭제**: 삭제 모드에서 기존 박스 내부 클릭
- **자동 라벨링**: Roboflow 기존 모델 또는 로컬 `best.pt`로 초안 생성
- **기존 라벨 검수**: 저장된 이미지를 불러와 수정 후 원본 라벨에 덮어쓰기

자동 라벨 결과는 정답이 아니므로 반드시 사람이 확인한 뒤 저장해야 합니다.

### 폴더 단위 자동 라벨링 (`auto_label.py`)

`studio.py`의 자동 라벨링은 이미지 한 장씩 UI로 돌리는 방식입니다. 폴더에 있는
이미지 여러 장을 한 번에 `best.pt`로 라벨링하려면:

```powershell
.\abcd\Scripts\python.exe auto_label.py --source path\to\images --split train
```

탐지된 타일을 YOLO 라벨로 `data/images/{split}`, `data/labels/{split}`에 저장합니다
(스튜디오의 자동 라벨링과 동일한 로직 재사용). `--confidence`로 신뢰도 임계값을,
`--allow-empty`로 타일이 하나도 안 잡힌 이미지도 저장할지 조정할 수 있습니다.

⚠️ 이렇게 만든 라벨은 모델 자신의 예측 결과라 **정답이 아닙니다**. `studio.py`의
"데이터 라벨링" 탭 → "기존 라벨 검수"에서 사람이 확인/수정해야 `evaluate.py`의
신뢰할 수 있는 정답으로 쓸 수 있습니다. 검수 없이 바로 `evaluate.py`를 돌리면
모델이 자기 자신의 예측과 비교되어 의미 없이 높은 점수만 나옵니다.

### 이미지 1장 검수 UI (`evaluate_ui.py`)

이미지 1장을 빠르게 탐지해보고, 결과를 즉석에서 고쳐서 새 학습 데이터셋으로
쌓고 싶을 때 쓰는 가벼운 페이지입니다 (`studio.py`보다 단순함).

```powershell
.\abcd\Scripts\python.exe evaluate_ui.py
```

기본으로 http://127.0.0.1:7861 에서 열립니다. 단계는 다음 순서로 진행합니다.

좌표를 직접 입력하는 대신 전부 **이미지를 클릭**해서 조작합니다.

1. **자르기 (선택)**: 이미지를 업로드/웹캠으로 올리면 그 즉시 2번 확대나 3번 탐지로
   바로 진행할 수 있습니다 — 자르기는 필수가 아닙니다. 특정 영역만 쓰고 싶을 때만,
   오른쪽 "여기를 클릭해서 자르기" 미리보기에서 자를 영역의 **첫 모서리**를 클릭한 뒤
   **반대쪽 모서리**를 클릭하세요. 빨간 사각형으로 확정되면 "자르기 적용"을 누릅니다.
   잘못 찍었으면 "모서리 선택 취소"로 첫 클릭만 취소할 수 있습니다.
2. **확대 (선택)**: 타일이 작아서 잘 안 보이면 배율을 올리고 "확대 적용"을 누릅니다.
   확대하지 않으면(기본 1.0배) 자른 이미지 그대로 다음 단계에 쓰입니다.
3. **탐지**: 신뢰도/입력 크기를 조정하고 "탐지 실행" → 탐지된 타일이 **박스별로 한
   행씩** 표에 나옵니다(클래스, 신뢰도, 좌표). 같은 위치에 절반 이상(IoU > 0.5) 겹치는
   박스가 있으면 신뢰도가 낮은 쪽을 **자동으로 지우고** 몇 개를 지웠는지 요약란에
   알려줍니다(클래스가 서로 달라도 위치가 겹치면 중복으로 봄).
4. **검수 (삭제/수정/추가)**: 표에서 잘못된 행을 삭제하거나(중복/오탐) 클래스 셀을
   더블클릭해서 고칩니다. 행을 클릭하면 결과 이미지에 그 박스 하나만 표시됩니다
   (같은 클래스가 여러 개여도 클릭한 것 하나만). "전체 보기"로 복귀. **모델이 놓친
   타일**은 **색상**과 **숫자**를 각각 드롭다운에서 고른 뒤(조커면 숫자는 무시됩니다)
   "탐지 결과" 이미지에서 두 모서리를 클릭하면(자르기와 같은 방식) 새 박스가 표에
   바로 추가됩니다. 색상 선택은 다음 박스를 추가할 때도 그대로 유지되니, 같은 색
   타일을 여러 개 추가할 때는 숫자만 바꿔가며 클릭하면 됩니다.
5. **저장**: split(train/val/test)을 고르고 "검수한 결과 데이터셋으로 저장"을 누르면,
   **1번에서 자른 이미지**(확대했다면 확대된 상태 그대로, 원본 전체 이미지가 아님)와
   YOLO 라벨(.txt)이 `C:\Users\SSAFY\workspace\rummikub_data\images\{split}`,
   `C:\Users\SSAFY\workspace\rummikub_data\labels\{split}`에 저장됩니다
   (`yolo_boardgame\data`와는 별도의 폴더입니다 — 나중에 학습에 합치려면 직접
   병합하거나 이 경로를 가리키는 `dataset.yaml`을 새로 만들면 됩니다).

### 폴더 일괄 검수 UI (`evaluate_ui_batch.py`)

`evaluate_ui.py`와 완전히 같은 기능(자르기/확대/탐지/검수/저장)이지만, 이미지를
한 장씩 업로드하는 대신 **폴더 경로를 입력**하면 그 폴더의 이미지를 순서대로 하나씩
보여주고, **저장할 때마다 자동으로 다음 이미지로 넘어갑니다**.

```powershell
.\abcd\Scripts\python.exe evaluate_ui_batch.py
```

기본으로 http://127.0.0.1:7862 에서 열립니다 (`evaluate_ui.py`와 동시에 켜둘 수 있게
포트가 다릅니다). "0. 폴더 불러오기"에 이미지 폴더 경로를 입력하고 "폴더 불러오기"를
누르면 시작합니다. 읽을 수 없는 파일은 자동으로 건너뜁니다. 저장하고 싶지 않은
이미지는 "건너뛰기"로 넘길 수 있습니다. 나머지 1~4단계(자르기/확대/탐지/검수)는
`evaluate_ui.py`와 완전히 동일합니다.

### 폴더 + 청테이프 평탄화 일괄 검수 UI (`evaluate_ui_flatten.py`)

`evaluate_ui_batch.py`와 같은 폴더 일괄 흐름에, **파란 테이프(청테이프) 모서리 자동 인식
+ 실측 가로/세로 비율로 평탄화**하는 단계가 추가된 버전입니다. 테이블 네 모서리에
파란 테이프를 붙이고 촬영한 사진 폴더를 다룰 때 씁니다 (`rummikub-webcam`
프로젝트에서 초록 테이프로 하던 평탄화와 같은 방식, 색만 파란색 기준).

```powershell
.\abcd\Scripts\python.exe evaluate_ui_flatten.py
```

기본으로 http://127.0.0.1:7863 에서 열립니다 (세 페이지를 동시에 켜둘 수 있게 포트가
다 다릅니다: `evaluate_ui.py` 7861, `evaluate_ui_batch.py` 7862, 이 페이지 7863).

이미지를 **불러오는 것**과 **평탄화하는 것**은 분리된 단계입니다 — 폴더를 불러오거나
"건너뛰기"/저장으로 다음 이미지로 넘어가도 평탄화는 자동 실행되지 않고 원본
미리보기만 먼저 보여줍니다. 평탄화는 매번 직접 "평탄화 실행"을 눌러야 합니다.

1. **폴더 불러오기**: 이미지 폴더 경로를 입력하고 "폴더 불러오기" → 원본 미리보기가
   나타납니다.
2. **평탄화**: 테이프로 표시한 영역의 가로/세로 실제 길이(cm)를 입력하고 "평탄화
   실행"을 누르면 파란 테이프 네 모서리를 찾아 입력한 비율로 평탄화합니다. 테이프가
   안 잡히면(⚠️ 표시) Hue 범위 슬라이더를 조정하고 다시 누르세요. 계속 실패하면
   원본이 그대로 다음 단계로 넘어가니, 1-1번 수동 자르기로 보완할 수 있습니다.
3. **1-1. 추가로 자르기 (선택)** → **2. 확대 (선택)** → **3. 탐지** → **4. 검수** →
   **5. 저장**: 나머지는 `evaluate_ui_batch.py`/`evaluate_ui.py`와 완전히 동일합니다.
   저장해도 자동으로 다음 이미지로 넘어가지 않으니, 준비되면 위쪽 "다음 이미지"를
   누르세요.

### 이미지 밝기 조정 UI (`brightness_ui.py`)

YOLO/탐지와 무관하게, 이미지 하나 올려서 밝기만 조정해보고 싶을 때 쓰는 완전히
독립된 페이지입니다.

```powershell
.\abcd\Scripts\python.exe brightness_ui.py
```

기본으로 http://127.0.0.1:7864 에서 열립니다. 이미지를 올리고 슬라이더로 밝기를
조정하면 바로 미리보기가 갱신됩니다. RGB에 단순히 값을 더하는 대신 HSV의 명도(V)만
조정해서 색상(Hue)이 틀어지지 않습니다. 결과 이미지는 우측 상단 다운로드 아이콘으로
저장할 수 있습니다.

### `rummikub_data` 데이터 증강 (`augment_dataset.py`)

`evaluate_ui*.py`로 `rummikub_data`에 저장한 이미지를 밝기/좌우반전 조합으로 불려서
학습 데이터를 늘리는 스크립트입니다.

```powershell
.\abcd\Scripts\python.exe augment_dataset.py
```

원본 이미지마다 밝기 `-25/0/+25` × `원본/좌우반전` 조합으로 **6장씩** 증강 이미지를
같은 split 폴더에 추가합니다. 라벨도 함께 변환됩니다 — 좌우반전이면 박스의
`center_x`를 `1 - center_x`로 뒤집고, 밝기만 바뀐 경우는 좌표가 그대로라 라벨을
복사합니다. 이미 증강된 파일은 파일명 접미사(`_bp25_flip` 등)로 식별해서 재실행해도
또 증강하지 않으니, 원본을 더 모은 뒤 다시 돌려도 안전합니다.

### 기존 라벨 검수/수정 UI (`review_labels_ui.py`)

`rummikub_data_2`처럼 **이미 라벨링이 끝난** 이미지+txt 쌍을 검수하고 고치는 페이지입니다.
`evaluate_ui*.py`와 달리 `best.pt`로 새로 탐지하지 않고, 있는 라벨을 그대로 불러와
이미지 위에 겹쳐 보여줍니다.

```powershell
.\abcd\Scripts\python.exe review_labels_ui.py
```

기본으로 http://127.0.0.1:7865 에서 열립니다. 데이터셋 폴더와 split(train/val/test)을
지정해 "불러오기"를 누르면 해당 split의 첫 이미지+라벨을 불러옵니다. 표에서 잘못된
행을 삭제하거나 클래스를 고치고, 놓친 타일은 이미지에서 두 모서리를 순서대로 클릭해
추가할 수 있습니다(색상/숫자는 별도 드롭다운으로 선택). **저장을 누르면 새 파일을
만드는 게 아니라 원본 라벨 txt를 그 자리에서 덮어씁니다.** "다음 이미지"는 저장과
분리되어 있어 저장 없이도 넘어갈 수 있습니다.

### 기존 공개 라벨 데이터

Roboflow Universe의 `rummikub-p8akb/2017` 버전을 가져와 YOLO Detect 형식으로
변환했습니다.

| 분할 | 이미지 | 박스 |
|---|---:|---:|
| Train | 1,095 | 11,249 |
| Validation | 107 | 1,179 |
| Test | 53 | 637 |
| 합계 | 1,255 | 13,065 |

원본 다각형 좌표는 외접 Bounding Box로 변환했고 `Black10` 같은 클래스명은
`black_10` 형식으로 통일했습니다. 다시 가져오려면 실행합니다.

```powershell
.\abcd\Scripts\python.exe import_roboflow.py
```

출처와 변환 기록은 [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) 및
`data/roboflow_import_manifest.json`에서 확인할 수 있습니다.

자동 라벨링에서 Roboflow를 선택하면 이미지가 API 서버로 전송됩니다. 로컬
`best.pt`와 MobileSAM은 이 PC의 GPU에서 실행됩니다.

## 3. 데이터 수집 기준

- 클래스: `black_1`~`black_13`, `blue_1`~`blue_13`,
  `orange_1`~`orange_13`, `red_1`~`red_13`, `joker`
- 이미지에 보이는 모든 타일을 빠짐없이 라벨링합니다.
- 박스는 타일 외곽에 밀착시킵니다.
- 조명, 배경, 거리, 각도, 그림자, 반사가 다른 장면을 포함합니다.
- 연속 프레임을 무작정 저장하지 말고 서로 다른 장면을 수집합니다.
- 같은 촬영 세션의 유사 이미지는 한 split에만 넣어 데이터 누수를 막습니다.
- 처음에는 클래스마다 최소 50개 이상을 목표로 하되, 검증 결과가 낮은
  클래스를 우선 보강합니다.

## 4. 데이터 검사

```powershell
.\abcd\Scripts\python.exe validate_dataset.py
```

이미지-라벨 쌍, 클래스 번호, 정규화 좌표 범위를 검사합니다.

## 5. 모델 파인튜닝 (기본 모델: `best.pt`)

이 저장소에는 루미큐브 타일로 이미 학습된 `best.pt`가 포함되어 있습니다(저장소에는
포함되지 않으므로 최초 준비 시 직접 복사해야 함 — [docs/readme.md](docs/readme.md) 참고).
COCO 사전학습 `yolo26n.pt`부터 새로 학습하지 않고, 이 `best.pt`를 시작점으로
공개 라벨과 직접 검수한 라벨을 추가 파인튜닝합니다.

```powershell
.\abcd\Scripts\python.exe train.py --epochs 100 --imgsz 960
```

결과 가중치는 다음에 저장됩니다.

```text
runs/detect/rummikub_tiles/weights/best.pt
```

중단된 학습 재개:

```powershell
.\abcd\Scripts\python.exe train.py `
  --resume runs\detect\rummikub_tiles\weights\last.pt
```

**클래스 순서 안내**: `best.pt`는 원래 다른 프로젝트에서 Roboflow 원본 라벨 텍스트
(`Black1`, `Black10`, ..., `Joker`, `Orange1`, ...)를 알파벳순으로 정렬한 순서로
학습되었습니다. 타일 번호 순서(1, 2, 3, ...)가 아니라 문자열 정렬 순서라 `black_1`
다음이 `black_10`인 것이 정상입니다. `rummikub/classes.py`의 `CLASS_NAMES`와
`dataset.yaml`을 이 순서에 맞춰뒀으므로 `data/labels`에 새로 저장되는 라벨은 자동으로
`best.pt`의 클래스 인덱스와 일치합니다. `validate_dataset.py`가 매번 `dataset.yaml`과
`CLASS_NAMES`의 순서 일치 여부를 검사하고, `evaluate.py`도 `best.pt`에 저장된 클래스
순서가 어긋나면 경고를 출력합니다.

### 평가 (mAP / 정밀도 / 재현율)

`validate_dataset.py`는 라벨 파일 형식만 검사하고 모델 성능은 측정하지 않습니다. 모델의
실제 탐지 성능(정밀도, 재현율, mAP)은 `evaluate.py`로 확인합니다. 학습을 따로 하지 않아도
저장소에 포함된 `best.pt`를 바로 평가할 수 있습니다.

```powershell
.\abcd\Scripts\python.exe evaluate.py --split test
```

`dataset.yaml`의 test split에 대해 Ultralytics 기본 평가를 실행하고 전체
정밀도/재현율/mAP50/mAP50-95와 클래스별 mAP50-95를 출력합니다. PR 곡선과 혼동 행렬은
`runs/detect/rummikub_tiles_eval/`에 저장됩니다. `--weights`로 다른 가중치를,
`--split val`로 검증 split을 평가할 수 있습니다.

## 6. 턴 판정 사용 조건

- 카메라는 테이블 위에 고정하고 턴 사이에 이동하지 않습니다.
- 기준 턴과 현재 턴에서 전체 테이블이 모두 보여야 합니다.
- 타일 세트 사이 간격을 타일 내부 간격보다 충분히 크게 둡니다.
- 플레이어별 최초 등록 완료 여부는 UI 체크박스로 관리합니다.
- 모델 오검출 가능성이 있으므로 대회 심판을 완전히 대체하는 용도가 아니라
  판정 보조 시스템으로 사용해야 합니다.

세부 규칙과 현재 자동 판정의 한계는 [docs/RULES_AND_LIMITATIONS.md](docs/RULES_AND_LIMITATIONS.md)를
참고하세요.

## 라이선스 주의

Ultralytics 저장소는 AGPL-3.0 및 Enterprise 라이선스 옵션을 제공합니다.
폐쇄형 또는 상용 서비스로 배포할 예정이라면 사용 방식에 맞는 라이선스를
반드시 확인해야 합니다.
