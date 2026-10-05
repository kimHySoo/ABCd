# Holography (A4 평면화) 개발 환경 설정

`holography.ipynb`는 `holography_data/` 폴더의 사진에서 A4 용지 모서리에 붙인 초록색 테이프를
검출하여, 용지를 위에서 내려다본 것처럼(top-down) 평면화(perspective rectification)합니다.

## 1. 가상환경 생성

프로젝트 루트(`rummikub-webcam/`)에서 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. 필요 라이브러리 설치

```powershell
pip install opencv-python numpy matplotlib jupyter ipykernel
```

| 라이브러리 | 용도 |
| --- | --- |
| `opencv-python` | 이미지 로드, HSV 색공간 변환, 컨투어 검출, 원근 변환(perspective transform) |
| `numpy` | 좌표 배열 연산 (꼭짓점 정렬, 거리 계산 등) |
| `matplotlib` | 원본/마스크/결과 이미지를 노트북에서 시각화 |
| `jupyter`, `ipykernel` | `holography.ipynb` 실행 및 VS Code/Jupyter 커널 등록 |

## 3. VS Code에서 커널 선택

노트북을 열고 우측 상단에서 방금 만든 `.venv` 커널을 선택한 뒤 셀을 순서대로 실행하세요.

## 4. 동작 개요

1. 이미지를 HSV로 변환 후 초록색 범위만 마스킹하여 테이프 4개의 컨투어를 찾음
2. 각 컨투어의 중심(모멘트)을 계산하고 좌상/우상/우하/좌하 순서로 정렬
3. A4 실제 비율(210:297mm)에 맞춘 목적 사각형을 정의
4. `cv2.getPerspectiveTransform` + `cv2.warpPerspective`로 탑다운 뷰 생성

테이프가 잘 검출되지 않으면 노트북 하단의 "튜닝 가이드"를 참고해 HSV 범위나 면적 비율을 조정하세요.
