"""Web UI to fix MELD GROUPING(어떤 타일이 어느 멜드에 속하는지)을 사람이 직접
검수/수정하는 도구. bbox/라벨은 이미 확정된 상태(`review_labels_ui.py`로 검수 끝난
`mc/data/holdout/annotations/*.json`)를 그대로 두고, 그 안의 "groups"(어떤 tile_id가
어느 group_id에 속하는지)만 고친다.

**이 도구가 필요한 이유(WORKLOG.md mc/ 32장/34장)**: 지금까지 annotation의 groups/decision은
`finalize_annotations.py`가 `pipeline.group_detections()`(평가하려는 그 알고리즘 자체)를
검수된 bbox/라벨에 돌려서 만든 것이라, "정답 그룹"이 독립적이지 않은 순환평가 문제가
있었다. 이 도구로 사람이 직접 그룹 경계를 확인/수정해야 진짜 정답이 된다.

**조작 방법**: 표에서 각 타일 행의 group_id 칸을 고치면 그룹이 바뀐다 -- 같은 번호를
주면 같은 그룹, 새 번호를 주면 새 그룹, 다른 타일과 번호를 맞바꾸면 그룹 이동이 된다.
표를 고친 뒤 "미리보기 갱신"으로 그룹 경계가 이미지에 어떻게 그려지는지, 그리고 각
그룹의 런/세트 유효성과 전체 PASS/FAIL이 어떻게 계산되는지(`rules.validate_group()`
그대로 재사용, 사람이 유효성 자체를 따로 입력할 필요 없음 -- 어떤 타일들이 묶였는지만
확인하면 유효성은 규칙에서 자동으로 나온다) 확인한 뒤 "저장"을 누른다.

사용법:
    python review_groups_ui.py
    (브라우저에서 뜨는 주소로 접속, 이미지 목록에서 하나씩 검수)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

MC_ROOT = Path(r"C:\Users\SSAFY\workspace\mc")
sys.path.insert(0, str(MC_ROOT))

from rules import validate_group, validate_run, validate_set  # noqa: E402


def _group_type(labels: list[str]) -> str:
    """mc/build_annotation_drafts.py, mc/finalize_annotations.py와 동일한 판정 순서
    (런 -> 세트)."""
    if validate_run(labels).valid:
        return "run"
    if validate_set(labels).valid:
        return "set"
    return "invalid"

ANNOTATION_DIR = MC_ROOT / "data" / "holdout" / "annotations"
DATA_ROOT = MC_ROOT / "data"

GROUP_COLORS = [
    (0, 200, 0), (220, 0, 0), (0, 0, 220), (0, 200, 200),
    (200, 0, 200), (200, 200, 0), (140, 0, 220), (0, 140, 220),
    (0, 120, 0), (120, 120, 0), (120, 0, 0), (0, 0, 120),
]

TABLE_HEADERS = ["tile_id", "label", "group_id"]
TABLE_DATATYPES = ["str", "str", "number"]


def _list_annotations() -> list[Path]:
    return sorted(ANNOTATION_DIR.glob("*.json"))


def _progress_text(paths: list[str], index: int) -> str:
    if not paths:
        return "annotation 파일이 없습니다."
    if index >= len(paths):
        return f"모든 이미지({len(paths)}장)를 처리했습니다."
    return f"{index + 1}/{len(paths)}: {Path(paths[index]).name}"


def _load_annotation(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _tiles_to_rows(record: dict) -> list[list]:
    """현재 annotation의 groups를 표 형태(타일 하나당 한 행, group_id 컬럼)로 변환.
    group_id는 원래 문자열("g0" 등)이지만 표에서 편집하기 쉽게 숫자만 남긴다."""
    group_of_tile: dict[str, str] = {}
    for g in record["groups"]:
        gid = str(g["group_id"]).lstrip("g")
        for tid in g["tile_ids"]:
            group_of_tile[tid] = gid
    rows = []
    for t in record["tiles"]:
        rows.append([t["id"], t["label"], group_of_tile.get(t["id"], "")])
    return rows


def _rows_to_groups(rows: list[list]) -> dict[str, list[str]]:
    """표(타일별 group_id)를 {group_id: [tile_id, ...]}로 역변환. group_id가 비어있는
    행(빈 문자열/None)은 어느 그룹에도 안 속한 것으로 보고 제외한다."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        if len(row) < 3:
            continue
        tile_id, _label, group_id = row[0], row[1], row[2]
        if group_id is None or str(group_id).strip() == "":
            continue
        key = str(group_id).strip()
        groups.setdefault(key, []).append(tile_id)
    return groups


def _draw(record: dict, rows: list[list]):
    image_path = DATA_ROOT / record["image"]
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()

    tiles_by_id = {t["id"]: t for t in record["tiles"]}
    groups = _rows_to_groups(rows)
    sorted_keys = sorted(groups.keys(), key=lambda g: (len(g), g))
    color_map = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(sorted_keys)}

    for group_id, tile_ids in groups.items():
        color = color_map[group_id]
        boxes = [tiles_by_id[tid]["bbox_xyxy"] for tid in tile_ids if tid in tiles_by_id]
        if not boxes:
            continue
        gx1 = min(b[0] for b in boxes)
        gy1 = min(b[1] for b in boxes)
        gx2 = max(b[2] for b in boxes)
        gy2 = max(b[3] for b in boxes)
        cv2.rectangle(image_rgb, (int(gx1), int(gy1)), (int(gx2), int(gy2)), color, 3)
        cv2.putText(
            image_rgb, f"group {group_id}", (int(gx1), max(0, int(gy1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        for tid in tile_ids:
            t = tiles_by_id.get(tid)
            if t is None:
                continue
            x1, y1, x2, y2 = t["bbox_xyxy"]
            cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(
                image_rgb, t["label"], (int(x1), max(0, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )
    return image_rgb.astype(np.uint8)


def _summarize(rows: list[list]) -> str:
    """각 그룹의 rules.validate_group() 결과와 전체 PASS/FAIL을 미리 보여준다 --
    사람이 유효성을 직접 판단할 필요 없이, 어느 타일이 어느 그룹인지만 맞추면 된다."""
    groups = _rows_to_groups(rows)
    ungrouped = [row[0] for row in rows if not str(row[2] if len(row) > 2 else "").strip()]

    lines = []
    reason_codes = []
    for group_id in sorted(groups.keys(), key=lambda g: (len(g), g)):
        tile_ids = groups[group_id]
        labels = [row[1] for row in rows if row[0] in tile_ids]
        result = validate_group(labels)
        status = "✅ OK" if result.valid else f"❌ FAIL({result.reason_code})"
        if not result.valid:
            reason_codes.append(result.reason_code)
        lines.append(f"- group {group_id}: {status} -- {labels}")

    if ungrouped:
        lines.append(f"\n⚠️ group_id가 비어있는 타일: {ungrouped} (그룹에 안 속한 상태로 저장됨 -- 확인 필요)")

    decision = "FAIL" if reason_codes else "PASS"
    lines.append(f"\n**전체 판정 미리보기: {decision}**  reason_codes={reason_codes}")
    return "\n".join(lines)


def load_by_index(index: int):
    paths = [str(p) for p in _list_annotations()]
    progress = _progress_text(paths, index)
    if not paths or index >= len(paths):
        return paths, index, None, [], "", progress

    record = _load_annotation(Path(paths[index]))
    rows = _tiles_to_rows(record)
    image = _draw(record, rows)
    summary = _summarize(rows)
    status = "이미 검수됨(group_reviewed=true)" if record.get("group_reviewed") else "미검수"
    return paths, index, image, rows, summary, f"{progress} -- {status}"


def refresh_preview(paths: list[str], index: int, rows: list[list]):
    if not paths or index >= len(paths):
        return None, ""
    record = _load_annotation(Path(paths[index]))
    image = _draw(record, rows)
    summary = _summarize(rows)
    return image, summary


def go_next(paths: list[str], index: int):
    return load_by_index(index + 1)


def go_prev(paths: list[str], index: int):
    return load_by_index(max(0, index - 1))


def save_groups(paths: list[str], index: int, rows: list[list]):
    if not paths or index >= len(paths):
        return "❌ 저장할 이미지가 없습니다."

    path = Path(paths[index])
    record = _load_annotation(path)
    tiles_by_id = {t["id"]: t for t in record["tiles"]}

    groups_dict = _rows_to_groups(rows)
    missing = [row[0] for row in rows if row[0] not in {tid for tids in groups_dict.values() for tid in tids}]
    if missing:
        return f"❌ 저장 안 함 -- group_id가 비어있는 타일이 있습니다: {missing}"

    # bbox min-x1 기준으로 그룹 순서를 정렬해서 g0, g1, ...로 다시 번호를 매긴다
    # (pipeline.group_detections()와 동일한 정렬 관례).
    def _group_min_x1(tile_ids: list[str]) -> float:
        return min(tiles_by_id[tid]["bbox_xyxy"][0] for tid in tile_ids if tid in tiles_by_id)

    ordered_keys = sorted(groups_dict.keys(), key=lambda g: _group_min_x1(groups_dict[g]))

    new_groups = []
    reason_codes = []
    for i, old_key in enumerate(ordered_keys):
        tile_ids = groups_dict[old_key]
        labels = [tiles_by_id[tid]["label"] for tid in tile_ids if tid in tiles_by_id]
        result = validate_group(labels)
        if not result.valid:
            reason_codes.append(result.reason_code)
        new_groups.append(
            {
                "group_id": f"g{i}",
                "tile_ids": tile_ids,
                "type": _group_type(labels),
                "valid": result.valid,
            }
        )

    decision = "FAIL" if reason_codes else "PASS"
    now = datetime.now().isoformat(timespec="seconds")

    record["groups"] = new_groups
    record["decision"] = decision
    record["reason_codes"] = reason_codes
    record["group_reviewed"] = True
    record["group_reviewed_at"] = now
    record.setdefault("notes", "")
    record["notes"] = (record["notes"] + f" | {now} 그룹 사람이 직접 검수함").strip(" |")

    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ `{path.name}` 저장 완료 -- decision={decision}, 그룹 {len(new_groups)}개"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Rummikub 그룹 검수") as demo:
        gr.Markdown(
            "## 멜드 그룹핑 검수/수정\n"
            "bbox/라벨은 이미 확정된 상태입니다. 표의 **group_id 칸만 고쳐서** 어떤 타일이\n"
            "어느 멜드에 속하는지 사람이 직접 맞춥니다. 같은 번호 = 같은 그룹, 새 번호 = 새\n"
            "그룹입니다. 유효성(런/세트/반칙)과 전체 PASS/FAIL은 그룹만 맞추면 규칙에서\n"
            "자동 계산되므로 따로 입력할 필요 없습니다."
        )

        paths_state = gr.State([])
        index_state = gr.State(0)

        with gr.Row():
            prev_button = gr.Button("이전 이미지")
            next_button = gr.Button("다음 이미지")
            load_button = gr.Button("처음부터 불러오기", variant="secondary")
        progress_output = gr.Markdown()

        image_output = gr.Image(label="그룹 미리보기 (굵은 테두리 = 그룹 경계)", interactive=False)

        table_output = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=TABLE_DATATYPES,
            row_count=0,
            column_count=(len(TABLE_HEADERS), "fixed"),
            type="array",
            label="타일별 group_id (이 칸만 고치세요)",
        )

        refresh_button = gr.Button("미리보기 갱신 (그룹/판정 다시 계산)")
        summary_output = gr.Markdown()

        save_button = gr.Button("저장", variant="primary")
        save_status_output = gr.Markdown()

        outputs = [paths_state, index_state, image_output, table_output, summary_output, progress_output]

        load_button.click(lambda: load_by_index(0), inputs=[], outputs=outputs)
        next_button.click(go_next, inputs=[paths_state, index_state], outputs=outputs)
        prev_button.click(go_prev, inputs=[paths_state, index_state], outputs=outputs)
        refresh_button.click(
            refresh_preview, inputs=[paths_state, index_state, table_output],
            outputs=[image_output, summary_output],
        )
        save_button.click(
            save_groups, inputs=[paths_state, index_state, table_output], outputs=[save_status_output]
        )

        demo.load(lambda: load_by_index(0), inputs=[], outputs=outputs)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_port=7866)
