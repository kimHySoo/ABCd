"""Minimal web UI: crop an image by clicking two corners, zoom in, run
best.pt, review/edit each detected box (delete duplicates, fix a wrong
class, or click two corners to draw a box for a missed tile), and export
the corrected result (the cropped photo) as a YOLO-format training sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from rummikub.annotations import draw_boxes, save_yolo_sample
from rummikub.classes import COLORS, CLASS_TO_ID, canonicalize_external_name
from rummikub.detector import detect_tiles


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "best.pt"
RUMMIKUB_DATA_ROOT = Path(r"C:\Users\SSAFY\workspace\rummikub_data_2")
SPLITS = ("train", "val", "test")
TABLE_HEADERS = ["#", "클래스", "신뢰도", "x1", "y1", "x2", "y2"]
TABLE_DATATYPES = ["number", "str", "number", "number", "number", "number", "number"]
DUPLICATE_IOU_THRESHOLD = 0.5  # boxes overlapping more than half (IoU) are treated as duplicates
MIN_BOX_SIZE = 2
JOKER_COLOR_OPTION = "joker"
NEW_BOX_COLOR_OPTIONS = [*COLORS, JOKER_COLOR_OPTION]
NEW_BOX_NUMBER_OPTIONS = [str(n) for n in range(1, 14)]


def combine_class_name(color: str, number: str) -> str:
    if color == JOKER_COLOR_OPTION:
        return "joker"
    return f"{color}_{int(number)}"


# --- crop: click two corners on the raw upload -----------------------------

def on_raw_image_change(image):
    """Cropping is optional: the raw upload is immediately usable as-is for
    zoom/detect, and applying a crop later just overwrites cropped/working."""
    if image is None:
        return None, None, None, None, None, None, ""
    clean = np.ascontiguousarray(image)
    return (
        clean,
        clean,
        None,
        None,
        clean,
        clean,
        "이미지를 두 번 클릭하면 자를 수 있습니다 (선택 사항). 자르지 않고 바로 2번 확대나 3번 탐지로 넘어가도 됩니다.",
    )


def _draw_marker(image, point):
    preview = image.copy()
    if point is not None:
        cv2.drawMarker(preview, point, (255, 200, 0), cv2.MARKER_CROSS, 24, 3)
    return preview


def _clamp_box(image, x1, y1, x2, y2) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def handle_crop_click(evt: gr.SelectData, original, pending):
    if original is None:
        return None, pending, None, "먼저 이미지를 업로드하세요."
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        return _draw_marker(original, pending), pending, None, "좌표를 읽지 못했습니다. 다시 클릭하세요."
    point = (int(index[0]), int(index[1]))
    if pending is None:
        return _draw_marker(original, point), point, None, f"첫 모서리 선택됨 {point}. 반대쪽 모서리를 클릭하세요."
    x1, y1, x2, y2 = _clamp_box(original, pending[0], pending[1], point[0], point[1])
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return original, None, None, "자를 영역이 너무 작아 취소했습니다. 다시 두 모서리를 클릭하세요."
    preview = original.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return preview, None, (x1, y1, x2, y2), f"영역 선택 완료: ({x1},{y1}) ~ ({x2},{y2}). '자르기 적용'을 누르세요."


def cancel_crop_pending(original, pending):
    if original is None:
        return None, None, "이미지가 없습니다."
    if pending is None:
        return original, None, "취소할 진행 중인 클릭이 없습니다."
    return original, None, "선택 중이던 첫 모서리를 취소했습니다."


def apply_crop(original, crop_box):
    if original is None:
        return None, None, None, "❌ 먼저 이미지를 업로드하세요."
    if crop_box is None:
        # No region selected -> treat "apply crop" as a no-op and keep the full image,
        # instead of wiping out the working image that upload already set.
        cropped = np.ascontiguousarray(original)
        return cropped, cropped, cropped, "자를 영역을 선택하지 않아 전체 이미지를 그대로 사용합니다."
    x1, y1, x2, y2 = _clamp_box(original, *crop_box)
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return None, None, None, "❌ 자를 영역이 너무 작습니다."
    cropped = np.ascontiguousarray(original[y1:y2, x1:x2])
    return cropped, cropped, cropped, f"✅ {x2 - x1}x{y2 - y1} 크기로 잘랐습니다."


# --- zoom --------------------------------------------------------------

def apply_zoom(cropped, factor):
    if cropped is None:
        return None, None
    factor = float(factor)
    if factor == 1.0:
        zoomed = cropped
    else:
        height, width = cropped.shape[:2]
        zoomed = cv2.resize(
            cropped, (max(1, int(width * factor)), max(1, int(height * factor))),
            interpolation=cv2.INTER_CUBIC,
        )
    return zoomed, zoomed


# --- detection -----------------------------------------------------------

def _iou(box_a: list[float], box_b: list[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_w = max(0.0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0.0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def remove_overlapping_duplicates(rows: list[list], threshold: float = DUPLICATE_IOU_THRESHOLD) -> tuple[list[list], int]:
    """Greedy NMS-style cleanup: when two boxes overlap more than `threshold`
    (IoU), keep only the higher-confidence one. Class-agnostic on purpose --
    a duplicate detection of the same physical tile can get a different
    guessed class, and Ultralytics' own NMS only dedupes within a class."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][2], reverse=True)
    kept_indices: list[int] = []
    for i in order:
        box = rows[i][3:7]
        if any(_iou(box, rows[j][3:7]) > threshold for j in kept_indices):
            continue
        kept_indices.append(i)

    kept_rows = []
    for new_index, i in enumerate(sorted(kept_indices), start=1):
        row = list(rows[i])
        row[0] = new_index
        kept_rows.append(row)
    return kept_rows, len(rows) - len(kept_rows)


def evaluate_image(image, weights, confidence, image_size):
    if image is None:
        return None, [], "먼저 1번에서 이미지를 업로드하세요."
    weights_path = Path(weights)
    if not weights_path.is_file():
        return None, [], f"❌ 가중치 파일을 찾을 수 없습니다: {weights_path}"
    try:
        tiles, annotated = detect_tiles(image, weights, confidence, int(image_size))
    except Exception as exc:  # noqa: BLE001 - surface any inference error in the UI
        return None, [], f"❌ 탐지 실패: {exc}"

    rows = []
    for index, tile in enumerate(tiles, start=1):
        try:
            class_name = canonicalize_external_name(tile.class_name)
        except ValueError:
            class_name = tile.class_name
        rows.append([index, class_name, round(tile.confidence, 3), tile.x1, tile.y1, tile.x2, tile.y2])

    raw_count = len(rows)
    rows, removed_count = remove_overlapping_duplicates(rows)
    annotated = draw_boxes(image, _rows_to_boxes(rows))

    summary_lines = [f"**탐지된 타일: {raw_count}개 → 중복 제거 후 {len(rows)}개**"]
    if removed_count:
        summary_lines.append(f"절반 이상(IoU > {DUPLICATE_IOU_THRESHOLD}) 겹치는 박스 {removed_count}개를 자동으로 지웠습니다(신뢰도 낮은 쪽 제거).")
    summary_lines.append("→ 놓친 타일은 결과 이미지를 두 번 클릭해서 직접 추가할 수 있습니다.")
    return annotated, rows, "  \n".join(summary_lines)


def _row_to_box(row: list) -> dict:
    _, class_name, _confidence, x1, y1, x2, y2 = row
    class_id = CLASS_TO_ID[canonicalize_external_name(str(class_name))]
    return {"class_id": class_id, "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}


def _rows_to_boxes(rows: list[list]) -> list[dict]:
    boxes = []
    for row in rows or []:
        try:
            boxes.append(_row_to_box(row))
        except (ValueError, KeyError, TypeError):
            continue
    return boxes


def filter_by_selected_class(evt: gr.SelectData, image, rows):
    if image is None or not rows or evt.index is None:
        return image, ""
    row_index = evt.index[0]
    if row_index >= len(rows):
        return image, ""
    selected_row = rows[row_index]
    try:
        boxes = [_row_to_box(selected_row)]
    except (ValueError, KeyError, TypeError):
        return image, f"❌ #{selected_row[0]} 박스 좌표/클래스를 읽을 수 없습니다."
    filtered = draw_boxes(image, boxes)
    return filtered, (
        f"**#{selected_row[0]} ({selected_row[1]}) 박스만 표시 중** — "
        "전체를 다시 보려면 '전체 보기'를 누르세요."
    )


def show_all_boxes(image, rows):
    if image is None:
        return image, ""
    boxes = _rows_to_boxes(rows)
    return draw_boxes(image, boxes), f"**전체 {len(boxes)}개 표시 중**"


# --- draw a new box for a missed tile: click two corners on the result image ---

def handle_box_click(evt: gr.SelectData, image, rows, pending, color, number):
    if image is None:
        return image, rows or [], pending, "먼저 탐지를 실행하세요."
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        return draw_boxes(image, _rows_to_boxes(rows), pending), rows or [], pending, "좌표를 읽지 못했습니다."
    point = (int(index[0]), int(index[1]))
    rows = [list(row) for row in (rows or [])]
    if pending is None:
        return (
            draw_boxes(image, _rows_to_boxes(rows), point),
            rows,
            point,
            f"첫 모서리 선택됨 {point}. 놓친 타일의 반대쪽 모서리를 클릭하세요.",
        )
    x1, y1, x2, y2 = _clamp_box(image, pending[0], pending[1], point[0], point[1])
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return draw_boxes(image, _rows_to_boxes(rows), None), rows, None, "박스가 너무 작아 취소했습니다. 다시 클릭하세요."
    class_name = combine_class_name(color, number)
    next_index = max((row[0] for row in rows), default=0) + 1
    rows.append([next_index, class_name, 1.0, x1, y1, x2, y2])
    return (
        draw_boxes(image, _rows_to_boxes(rows), None),
        rows,
        None,
        f"✅ #{next_index} ({class_name}) 박스를 추가했습니다. 현재 {len(rows)}개.",
    )


def cancel_box_pending(image, rows, pending):
    if image is None:
        return None, rows or [], None, "이미지가 없습니다."
    if pending is None:
        return draw_boxes(image, _rows_to_boxes(rows), None), rows or [], None, "취소할 진행 중인 클릭이 없습니다."
    return draw_boxes(image, _rows_to_boxes(rows), None), rows or [], None, "선택 중이던 첫 모서리를 취소했습니다."


def redraw_from_table(image, rows):
    """Keep the result image in sync with the table no matter how the table
    changed (box added by click, row deleted, cell edited by hand, etc.) --
    the table is the single source of truth, the image is just its render."""
    if image is None:
        return None
    return draw_boxes(image, _rows_to_boxes(rows))


# --- save ------------------------------------------------------------------

def save_dataset(image, rows, split):
    if image is None:
        return "❌ 저장할 이미지가 없습니다 (자르기를 먼저 적용하세요)."
    if not rows:
        return "❌ 저장할 박스가 없습니다 (표가 비어 있습니다)."

    boxes = []
    bad_rows = []
    for row in rows:
        try:
            boxes.append(_row_to_box(row))
        except (ValueError, KeyError, TypeError):
            bad_rows.append(row)
    if bad_rows:
        return f"❌ 처리할 수 없는 행이 있습니다 (클래스명/좌표 확인): {bad_rows}"

    try:
        image_path, label_path = save_yolo_sample(image, boxes, split, root=RUMMIKUB_DATA_ROOT)
    except ValueError as exc:
        return f"❌ 저장 실패: {exc}"

    return (
        f"✅ 저장 완료 ({len(boxes)}개 박스)\n\n"
        f"- 이미지: `{image_path}`\n"
        f"- 라벨: `{label_path}`"
    )


def build_app(default_weights: Path = DEFAULT_WEIGHTS) -> gr.Blocks:
    with gr.Blocks(title="Rummikub 타일 탐지 검수") as demo:
        gr.Markdown(
            "## 자르기 → 확대 → 탐지 → 검수/수정 → 데이터셋 저장\n"
            "좌표를 직접 입력하지 않고 전부 **이미지를 클릭**해서 조작합니다. 자르기와 "
            "놓친 타일 추가는 두 모서리를 순서대로 클릭하면 됩니다. 표에서 중복/오탐 "
            "행을 삭제하거나 클래스를 고친 뒤, **자른 사진 기준으로** "
            f"`{RUMMIKUB_DATA_ROOT}`에 YOLO 데이터셋으로 저장할 수 있습니다."
        )

        working_state = gr.State(None)  # image actually used for detect/save (crop, optionally zoomed)
        cropped_state = gr.State(None)  # crop result at 1x, kept so zoom can be re-applied from scratch
        raw_original_state = gr.State(None)  # clean copy of the raw upload, redrawn from on every click
        crop_pending_state = gr.State(None)
        crop_box_state = gr.State(None)

        gr.Markdown(
            "### 1. 이미지 업로드 & 자르기 (선택)\n"
            "이미지를 올리면 그 즉시 2번 확대나 3번 탐지로 바로 진행할 수 있습니다 — "
            "자르기는 필수가 아닙니다. 특정 영역만 쓰고 싶을 때만, 오른쪽 미리보기를 "
            "클릭해서 자를 영역의 **첫 모서리**를 찍고 반대쪽 모서리를 다시 클릭한 뒤 "
            "'자르기 적용'을 누르세요."
        )
        with gr.Row():
            raw_image_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="원본 이미지 업로드")
            crop_click_preview = gr.Image(type="numpy", interactive=True, label="여기를 클릭해서 자르기 (모서리 2번 클릭)")
        with gr.Row():
            crop_button = gr.Button("자르기 적용", variant="primary")
            crop_cancel_button = gr.Button("모서리 선택 취소")
        crop_status_output = gr.Markdown()
        cropped_preview = gr.Image(type="numpy", label="자른 이미지", interactive=False)

        raw_image_input.change(
            on_raw_image_change,
            inputs=[raw_image_input],
            outputs=[
                raw_original_state,
                crop_click_preview,
                crop_pending_state,
                crop_box_state,
                cropped_state,
                working_state,
                crop_status_output,
            ],
        )
        crop_click_preview.select(
            handle_crop_click,
            inputs=[raw_original_state, crop_pending_state],
            outputs=[crop_click_preview, crop_pending_state, crop_box_state, crop_status_output],
        )
        crop_cancel_button.click(
            cancel_crop_pending,
            inputs=[raw_original_state, crop_pending_state],
            outputs=[crop_click_preview, crop_pending_state, crop_status_output],
        )
        crop_button.click(
            apply_crop,
            inputs=[raw_original_state, crop_box_state],
            outputs=[cropped_state, working_state, cropped_preview, crop_status_output],
        )

        gr.Markdown("### 2. 확대 (선택)")
        with gr.Row():
            zoom_slider = gr.Slider(1.0, 4.0, value=1.0, step=0.1, label="확대 배율")
            zoom_button = gr.Button("확대 적용")
        zoomed_preview = gr.Image(type="numpy", label="분석에 사용될 이미지", interactive=False)

        zoom_button.click(
            apply_zoom,
            inputs=[cropped_state, zoom_slider],
            outputs=[working_state, zoomed_preview],
        )

        gr.Markdown("### 3. 탐지")
        with gr.Row():
            weights_input = gr.Textbox(value=str(default_weights), label="모델 가중치 경로")
            confidence_input = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="신뢰도 임계값")
            image_size_input = gr.Slider(640, 1280, value=960, step=32, label="입력 크기")
        run_button = gr.Button("탐지 실행", variant="primary")
        annotated_output = gr.Image(type="numpy", interactive=True, label="탐지 결과 (클릭해서 놓친 타일 추가)")
        summary_output = gr.Markdown()

        gr.Markdown(
            "### 4. 검수 (표에서 직접 삭제/수정 + 클릭으로 추가/필터)\n"
            "각 행은 탐지된 타일 1개입니다. **표에서 행을 클릭**하면 위 결과 이미지에 "
            "그 박스 하나만 표시됩니다(같은 클래스가 여러 개여도 클릭한 것 하나만). "
            "겹치는 중복 박스는 위 안내에 뜬 번호를 보고 행 오른쪽의 삭제 버튼으로 "
            "지우세요. 클래스가 틀렸으면 셀을 더블클릭해서 고치세요.\n\n"
            "**놓친 타일이 있으면**: 아래에서 색상/숫자를 고른 뒤, 위 '탐지 결과' 이미지를 "
            "두 번 클릭(첫 모서리 → 반대쪽 모서리)하면 새 박스가 표에 추가됩니다. "
            "색상은 다음 박스를 추가할 때도 마지막 선택이 그대로 유지됩니다."
        )
        with gr.Row():
            new_box_color_input = gr.Dropdown(NEW_BOX_COLOR_OPTIONS, value=NEW_BOX_COLOR_OPTIONS[0], label="색상")
            new_box_number_input = gr.Dropdown(NEW_BOX_NUMBER_OPTIONS, value=NEW_BOX_NUMBER_OPTIONS[0], label="숫자")
            add_box_cancel_button = gr.Button("박스 추가 취소")
        table_output = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=TABLE_DATATYPES,
            row_count=0,  # plain int -> dynamic rows, so the per-row delete control shows up
            # column_count as a (count, "fixed") tuple keeps the 7 columns from being
            # inserted/deleted while still leaving cell values editable (unlike
            # static_columns, which also blocks editing the values themselves).
            column_count=(len(TABLE_HEADERS), "fixed"),
            interactive=True,
            type="array",
            label="탐지된 타일 목록 (편집 가능, 행 클릭 시 그 박스 하나만 표시)",
        )
        with gr.Row():
            filter_status_output = gr.Markdown()
            show_all_button = gr.Button("전체 보기")

        box_pending_state = gr.State(None)

        run_button.click(
            evaluate_image,
            inputs=[working_state, weights_input, confidence_input, image_size_input],
            outputs=[annotated_output, table_output, summary_output],
        )
        annotated_output.select(
            handle_box_click,
            inputs=[working_state, table_output, box_pending_state, new_box_color_input, new_box_number_input],
            outputs=[annotated_output, table_output, box_pending_state, summary_output],
        )
        add_box_cancel_button.click(
            cancel_box_pending,
            inputs=[working_state, table_output, box_pending_state],
            outputs=[annotated_output, table_output, box_pending_state, summary_output],
        )
        table_output.select(
            filter_by_selected_class,
            inputs=[working_state, table_output],
            outputs=[annotated_output, filter_status_output],
        )
        show_all_button.click(
            show_all_boxes,
            inputs=[working_state, table_output],
            outputs=[annotated_output, filter_status_output],
        )
        # Safety net: whenever the table's data changes for ANY reason (row added by
        # click, row deleted, cell edited by hand), redraw the result image from it so
        # the two can never drift out of sync with each other.
        table_output.change(
            redraw_from_table,
            inputs=[working_state, table_output],
            outputs=[annotated_output],
        )

        gr.Markdown("### 5. 데이터셋으로 저장 (자른 사진 기준)")
        with gr.Row():
            split_input = gr.Radio(list(SPLITS), value="train", label="저장할 split")
            save_button = gr.Button("검수한 결과 데이터셋으로 저장", variant="primary")
        save_status_output = gr.Markdown()

        save_button.click(
            save_dataset,
            inputs=[working_state, table_output, split_input],
            outputs=[save_status_output],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review UI for evaluating and correcting Rummikub detections")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app(args.weights).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
    )
