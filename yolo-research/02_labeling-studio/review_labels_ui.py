"""Web UI to review and fix labels that already exist in a YOLO dataset
folder (e.g. rummikub_data_2): load an image/label pair, overlay the
existing boxes, edit them with the same table/click tools as the other
pages (delete a row, fix a class, click two corners to add a missed box),
then save -- which overwrites the original label file in place. No YOLO
model is involved; this only edits ground-truth labels that already exist.

Default root now points at the Rummikub facilitator's (mc/) holdout review
set: build_yolo_review_dataset.py (in mc/) converts mc/data's captured
holdout images + YOLOv8s draft predictions into this images/<split>+
labels/<split> layout, using data/{init,middle,last} as the "split" names
instead of train/val/test (see REVIEW_SPLITS below -- kept local to this
file so evaluate_ui.SPLITS, shared by the other pages, stays untouched).
Pass --root to point at a different dataset (e.g. rummikub_data_2) instead."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr

from evaluate_ui import (
    NEW_BOX_COLOR_OPTIONS,
    NEW_BOX_NUMBER_OPTIONS,
    TABLE_DATATYPES,
    TABLE_HEADERS,
    _row_to_box,
    cancel_box_pending,
    filter_by_selected_class,
    handle_box_click,
    redraw_from_table,
    show_all_boxes,
)
from rummikub.annotations import draw_boxes, normalize_box
from rummikub.classes import CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ROOT = Path(r"C:\Users\SSAFY\workspace\mc\data\holdout\yolo_review")
REVIEW_SPLITS = ("init", "middle", "last")


def _list_pairs(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    if not image_dir.is_dir():
        return []
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _label_path_for(root: Path, split: str, image_path: Path) -> Path:
    return root / "labels" / split / f"{image_path.stem}.txt"


def _progress_text(paths: list[Path], index: int) -> str:
    if not paths:
        return "불러온 이미지가 없습니다."
    if index >= len(paths):
        return f"모든 이미지({len(paths)}장)를 처리했습니다."
    return f"{index + 1}/{len(paths)}: {paths[index].name}"


def load_pair(image_path: Path, label_path: Path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None, None, [], f"이미지를 읽을 수 없습니다: {image_path.name}"
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    rows = []
    if label_path.is_file():
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                center_x, center_y, box_width, box_height = (float(value) for value in parts[1:5])
            except ValueError:
                continue
            x1 = (center_x - box_width / 2) * width
            y1 = (center_y - box_height / 2) * height
            x2 = (center_x + box_width / 2) * width
            y2 = (center_y + box_height / 2) * height
            class_name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"class_{class_id}"
            rows.append([index, class_name, 1.0, round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])

    boxes = []
    for row in rows:
        try:
            boxes.append(_row_to_box(row))
        except (ValueError, KeyError, TypeError):
            continue
    annotated = draw_boxes(image_rgb, boxes)
    return image_rgb, annotated, rows, f"박스 {len(rows)}개 불러옴: {image_path.name}"


def _go_to(root: Path, split: str, paths: list[str], index: int):
    path_objs = [Path(p) for p in paths]
    progress = _progress_text(path_objs, index)
    if not paths or index >= len(paths):
        return paths, index, None, None, None, [], progress
    image_path = path_objs[index]
    label_path = _label_path_for(root, split, image_path)
    image_rgb, annotated, rows, status = load_pair(image_path, label_path)
    return paths, index, image_rgb, annotated, str(label_path), rows, f"{progress} — {status}"


def load_split(root_str: str, split: str):
    root = Path(root_str) if root_str else DEFAULT_ROOT
    paths = _list_pairs(root, split)
    if not paths:
        return [], -1, None, None, None, [], f"❌ {root / 'images' / split}에 이미지가 없습니다."
    return _go_to(root, split, [str(p) for p in paths], 0)


def go_to_next_pair(root_str: str, split: str, paths: list[str], index: int):
    root = Path(root_str) if root_str else DEFAULT_ROOT
    if not paths:
        return paths, index, None, None, None, [], "❌ 먼저 폴더를 불러오세요."
    return _go_to(root, split, paths, index + 1)


def save_overwrite(image, rows, label_path_str):
    if image is None:
        return "❌ 저장할 이미지가 없습니다."
    if not label_path_str:
        return "❌ 대상 라벨 경로를 알 수 없습니다 (먼저 이미지를 불러오세요)."

    boxes = []
    bad_rows = []
    for row in rows or []:
        try:
            boxes.append(_row_to_box(row))
        except (ValueError, KeyError, TypeError):
            bad_rows.append(row)
    if bad_rows:
        return f"❌ 처리할 수 없는 행이 있습니다 (클래스명/좌표 확인): {bad_rows}"

    height, width = image.shape[:2]
    try:
        lines = [normalize_box(box, width, height) for box in boxes]
    except ValueError as exc:
        return f"❌ 저장 실패: {exc}"

    label_path = Path(label_path_str)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return f"✅ `{label_path.name}`에 박스 {len(boxes)}개를 덮어썼습니다."


def build_app(default_root: Path = DEFAULT_ROOT) -> gr.Blocks:
    with gr.Blocks(title="Rummikub 라벨 검수") as demo:
        gr.Markdown(
            "## 기존 라벨 검수/수정\n"
            "폴더의 이미지와 라벨을 겹쳐서 보여줍니다. 표에서 잘못된 행을 삭제하거나 "
            "클래스를 고치고, 놓친 타일은 이미지에서 두 모서리를 순서대로 클릭해 "
            "추가하세요. **저장하면 원본 라벨 파일을 덮어씁니다.**"
        )

        paths_state = gr.State([])
        current_index_state = gr.State(-1)
        image_state = gr.State(None)
        label_path_state = gr.State(None)
        box_pending_state = gr.State(None)

        with gr.Row():
            root_path_input = gr.Textbox(label="데이터셋 폴더", value=str(default_root))
            split_input = gr.Radio(list(REVIEW_SPLITS), value=REVIEW_SPLITS[0], label="split")
            load_button = gr.Button("불러오기", variant="primary")
            next_button = gr.Button("다음 이미지")
        folder_status_output = gr.Markdown()

        annotated_output = gr.Image(label="박스 검수 (클릭해서 놓친 타일 추가)", interactive=False)
        summary_output = gr.Markdown()

        with gr.Row():
            new_box_color_input = gr.Dropdown(NEW_BOX_COLOR_OPTIONS, value=NEW_BOX_COLOR_OPTIONS[0], label="추가할 박스 색상")
            new_box_number_input = gr.Dropdown(NEW_BOX_NUMBER_OPTIONS, value=NEW_BOX_NUMBER_OPTIONS[0], label="추가할 박스 숫자")
            cancel_box_button = gr.Button("진행 중인 박스 취소")

        table_output = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=TABLE_DATATYPES,
            row_count=0,
            column_count=(len(TABLE_HEADERS), "fixed"),
            type="array",
            label="박스 목록 (행 삭제/클래스 수정 가능)",
        )
        with gr.Row():
            show_all_button = gr.Button("전체 보기")
            filter_status_output = gr.Markdown()

        save_button = gr.Button("저장 (라벨 파일 덮어쓰기)", variant="primary")
        save_status_output = gr.Markdown()

        load_outputs = [
            paths_state,
            current_index_state,
            image_state,
            annotated_output,
            label_path_state,
            table_output,
            folder_status_output,
        ]
        load_button.click(load_split, inputs=[root_path_input, split_input], outputs=load_outputs)
        next_button.click(
            go_to_next_pair,
            inputs=[root_path_input, split_input, paths_state, current_index_state],
            outputs=load_outputs,
        )

        annotated_output.select(
            handle_box_click,
            inputs=[image_state, table_output, box_pending_state, new_box_color_input, new_box_number_input],
            outputs=[annotated_output, table_output, box_pending_state, summary_output],
        )
        cancel_box_button.click(
            cancel_box_pending,
            inputs=[image_state, table_output, box_pending_state],
            outputs=[annotated_output, table_output, box_pending_state, summary_output],
        )
        table_output.change(redraw_from_table, inputs=[image_state, table_output], outputs=[annotated_output])
        table_output.select(
            filter_by_selected_class,
            inputs=[image_state, table_output],
            outputs=[annotated_output, filter_status_output],
        )
        show_all_button.click(
            show_all_boxes, inputs=[image_state, table_output], outputs=[annotated_output, filter_status_output]
        )

        save_button.click(save_overwrite, inputs=[image_state, table_output, label_path_state], outputs=[save_status_output])

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--port", type=int, default=7865)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app(args.root)
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main()
