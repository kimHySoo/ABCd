# GitLab 커밋 기록

SSAFY 반출 시 결과물만 압축 파일로 제공되어 커밋 이력이 함께 넘어오지 않기 때문에, 반출 전에 원본 GitLab 저장소의 기록을 남겨 둡니다.

- 저장소: `lab.ssafy.com/s15-webmobile3-sub1/S15P11A402`
- 조회일: 2026-10-05
- 기간: 2026-07-14 ~ 2026-08-13
- 범위: 전체 브랜치, 커밋 220개 (merge 제외 155개)

## 1. 작성자별 커밋 수

`git shortlog -sn --no-merges --all` 결과 (이메일 제외)

```text
   108	junseo
    20	박건우
    11	김형수
     4	A402_박건우
     3	Codex
     3	kjh
     2	kimHySoo
     2	임다빈
     1	junseo
     1	박준서
```

같은 사람의 계정 이름을 합치고, 커밋이 수정한 경로 기준으로 정리한 표입니다. 커밋 하나가 여러 경로를 수정하면 각 경로에 모두 셉니다.

| 작성자 | 계정 이름 | 커밋 | 주로 작업한 경로 |
| --- | --- | ---: | --- |
| 박준서 | junseo, 박준서 | 110 | `inte-server` 55, `boardgame-frontend` 50, `exec` 5, `table/wakeword` 3 |
| 박건우 | 박건우, A402_박건우 | 24 | `cabinet/controller` 16, `cabinet-controller`(이전 경로) 4, `cabinet/unit` 3, `docs` 2 |
| **김형수** | 김형수, kimHySoo | **13** | **`table/mc` 11**, `table` 설정 4, `README.md` 2 |
| 김지호 | kjh | 3 | 루트 파일 2, `cabinet-controller` 1 |
| 임다빈 | 임다빈 | 2 | `table/qna-server` 1, 루트 파일 1 |
| (AI 에이전트) | Codex | 3 | `docs`, `cabinet` |

커밋 수는 커밋 습관(작게 자주 / 크게 한 번), 브랜치 정리 방식에 따라 크게 달라지므로 기여량을 그대로 나타내지는 않습니다.

## 2. `table/mc` (루미큐브 AI 진행자) 작성자

`git log --all --no-merges --format='%an' -- table/mc`

```text
     11 김형수
```

AI 진행자 코드가 있는 `table/mc` 경로의 커밋은 11개 모두 본인 작성입니다.

## 3. 본인 커밋 목록

| 커밋 | 날짜 | 브랜치 | 메시지 | 주요 파일 |
| --- | --- | --- | --- | --- |
| `79c60c1` | 2026-08-13 13:10 | master | docs:회고록 수정 | `README.md` |
| `b6db4da` | 2026-08-13 13:09 | master | docs:회고록 추가 | `README.md` |
| `332ce30` | 2026-08-06 08:36 | feat/facilitator | fix:설치파일 누락 | `install_jetson.sh` |
| `cfb8351` | 2026-08-05 15:06 | master | mc 기능 추가 | `table/mc` 전체 (파이프라인, 규칙, MQTT, 설치 스크립트, 가중치) |
| `698e4ad` | 2026-08-05 11:03 | feat/facilitator | docs:gitignore에 테스트 폴더 추가 | `table/mc/.gitignore` |
| `bfd6285` | 2026-08-05 10:52 | feat/facilitator | fix:파일 위치 재구성 | `table/` → `table/mc/` 이동, `SETUP.md`, `install_jetson.sh` |
| `55191f3` | 2026-08-05 09:22 | feat/facilitator | fix:카메라 초기 프레임 드랍 횟수 변경, yolo 임계치 수정 | `pipeline.py` |
| `2c778c0` | 2026-08-04 15:05 | feat/facilitator | feat: 루미큐브 틀린 그룹 리스트 전달 | `mqtt_handler.py`, `pipeline.py` |
| `1cd441a` | 2026-08-04 13:39 | feat/facilitator | docs:test 결과물 삭제 | `jt_result/` 삭제 |
| `16c8c2f` | 2026-08-04 13:37 | feat/facilitator | fix: 카메라 인덱스 불일치시 자동 삭제 | `mqtt_handler.py`, `pipeline.py`, `rules.py` |
| `056668b` | 2026-08-04 09:32 | feat/facilitator | feat:table/mc 기능 생성 | `pipeline.py`, `rules.py`, `mqtt_handler.py`, `state.py`, `process_lock.py`, 가중치 |
| `1f24107` | 2026-08-02 21:40 | feat/facilitator | fix: 타일별 세트 군집 알고리즘 수정, yolo 모델 변경 | `pipeline.py`, YOLOv8n·YOLO11n 가중치 |
| `fbe9eb4` | 1970-01-01 09:02 | feat/facilitator | fix:카메라 초기 프레임 드랍 횟수 변경 | `pipeline_jt.py` |

- `fbe9eb4`의 1970-01-01은 시계가 동기화되지 않은 Jetson에서 커밋해 생긴 날짜입니다.
- `feat/facilitator` 브랜치는 2026-08-05에 master로 merge되었습니다(`5dfbf65`).
- 이 표의 커밋 해시는 원본 GitLab 저장소 기준이며, 반출본에는 남지 않습니다.
