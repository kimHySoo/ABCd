"""Same review pipeline as evaluate_ui.py (crop -> zoom -> detect -> review
-> save), but driven by a folder of images instead of one manual upload:
point it at a folder and work through each image. Saving does NOT advance
by itself -- you can crop a different region of the same image and save
again as a separate sample; click "다음 이미지" when you're done with it."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr

from evaluate_ui import (
    DEFAULT_WEIGHTS,
    NEW_BOX_COLOR_OPTIONS,
    NEW_BOX_NUMBER_OPTIONS,
    RUMMIKUB_DATA_ROOT,
    SPLITS,
    TABLE_DATATYPES,
    TABLE_HEADERS,
    apply_crop,
    apply_zoom,
    cancel_box_pending,
    cancel_crop_pending,
    evaluate_image,
    filter_by_selected_class,
    handle_box_click,
    handle_crop_click,
    on_raw_image_change,
    redraw_from_table,
    save_dataset,
    show_all_boxes,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _progress_text(file_list: list[str], index: int) -> str:
    if not file_list:
        return "불러온 이미지가 없습니다."
    if index >= len(file_list):
        return f"✅ 폴더의 이미지 {len(file_list)}장을 모두 처리했습니다."
    return f"{index + 1}/{len(file_list)}: {Path(file_list[index]).name}"


def _find_next_readable(file_list: list[str], start_index: int):
    index = start_index
    while index < len(file_list):
        image = _read_image(Path(file_list[index]))
        if image is not None:
            return index, image
        index += 1
    return index, None


def load_folder(folder_path: str):
    if not folder_path:
        return [], -1, None, "❌ 폴더 경로를 입력하세요."
    folder = Path(folder_path)
    if not folder.is_dir():
        return [], -1, None, f"❌ 폴더를 찾을 수 없습니다: {folder}"
    files = _list_images(folder)
    if not files:
        return [], -1, None, f"❌ 이미지 파일이 없습니다: {folder}"
    file_list = [str(p) for p in files]
    index, image = _find_next_readable(file_list, 0)
    if image is None:
        return file_list, index, None, f"❌ 읽을 수 있는 이미지가 없습니다: {folder}"
    return file_list, index, image, f"✅ {len(files)}장을 불러왔습니다. {_progress_text(file_list, index)}"


def advance_to_next(file_list: list[str], current_index: int):
    next_index, image = _find_next_readable(file_list, current_index + 1)
    return file_list, next_index, image, _progress_text(file_list, next_index)


def go_to_next_image(file_list, index):
    """Moves on regardless of whether the current image was saved yet --
    save and "next" are independent actions so the same image can be
    cropped/detected/saved multiple times (once per region) before moving on."""
    file_list, next_index, next_image, progress = advance_to_next(file_list, index)
    return f"다음 이미지로 이동했습니다. {progress}", next_image, file_list, next_index


def clear_detection_outputs():
    return None, [], ""


def build_app(default_weights: Path = DEFAULT_WEIGHTS) -> gr.Blocks:
    with gr.Blocks(title="Rummikub 타일 탐지 검수 (폴더 일괄)") as demo:
        gr.Markdown(
            "## 폴더 일괄 검수: 자르기 → 확대 → 탐지 → 검수/수정 → 저장 → (준비되면) 다음 이미지\n"
            "폴더를 불러오면 이미지를 한 장씩 순서대로 보여줍니다. **저장과 다음 이미지는 "
            "분리된 동작입니다** — 저장해도 자동으로 넘어가지 않으니, 같은 이미지에서 "
            "다른 영역을 다시 잘라 별도 샘플로 여러 번 저장한 뒤 준비되면 '다음 이미지'를 "
            "누르세요. 나머지 조작은 1장짜리 검수 페이지(`evaluate_ui.py`)와 동일하게 전부 "
            f"클릭으로 합니다. 결과는 매번 `{RUMMIKUB_DATA_ROOT}`에 저장됩니다."
        )

        file_list_state = gr.State([])
        current_index_state = gr.State(-1)

        gr.Markdown("### 0. 폴더 불러오기")
        with gr.Row():
            folder_path_input = gr.Textbox(label="이미지 폴더 경로", placeholder=r"C:\path\to\images")
            load_folder_button = gr.Button("폴더 불러오기", variant="primary")
            next_button = gr.Button("다음 이미지")
        folder_status_output = gr.Markdown()

        working_state = gr.State(None)  # image actually used for detect/save (crop, optionally zoomed)
        cropped_state = gr.State(None)  # crop result at 1x, kept so zoom can be re-applied from scratch
        raw_original_state = gr.State(None)  # clean copy of the current image, redrawn from on every click
        crop_pending_state = gr.State(None)
        crop_box_state = gr.State(None)

        gr.Markdown(
            "### 1. 자르기 (선택)\n"
            "현재 이미지가 그 즉시 2번 확대나 3번 탐지에 바로 쓰입니다 — 자르기는 필수가 "
            "아닙니다. 특정 영역만 쓰고 싶을 때만, 오른쪽 미리보기를 클릭해서 자를 "
            "영역의 **첫 모서리**를 찍고 반대쪽 모서리를 다시 클릭한 뒤 '자르기 적용'을 "
            "누르세요."
        )
        with gr.Row():
            raw_image_input = gr.Image(type="numpy", interactive=False, label="현재 이미지 (폴더에서 자동으로 채워짐)")
            crop_click_preview = gr.Image(type="numpy", interactive=True, label="여기를 클릭해서 자르기 (모서리 2번 클릭)")
        with gr.Row():
            crop_button = gr.Button("자르기 적용", variant="primary")
            crop_cancel_button = gr.Button("모서리 선택 취소")
        crop_status_output = gr.Markdown()
        cropped_preview = gr.Image(type="numpy", label="자른 이미지", interactive=False)

        gr.Markdown("### 2. 확대 (선택)")
        with gr.Row():
            zoom_slider = gr.Slider(1.0, 4.0, value=1.0, step=0.1, label="확대 배율")
            zoom_button = gr.Button("확대 적용")
        zoomed_preview = gr.Image(type="numpy", label="분석에 사용될 이미지", interactive=False)

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
            "겹치는 중복 박스는 자동으로 지워집니다. 클래스가 틀렸으면 셀을 더블클릭해서 "
            "고치세요.\n\n"
            "**놓친 타일이 있으면**: 아래에서 색상/숫자를 고른 뒤, 위 '탐지 결과' 이미지를 "
            "두 번 클릭(첫 모서리 → 반대쪽 모서리)하면 새 박스가 표에 추가됩니다. 색상은 "
            "다음 박스를 추가할 때도(그리고 다음 이미지로 넘어가도) 마지막 선택이 그대로 "
            "유지됩니다."
        )
        with gr.Row():
            new_box_color_input = gr.Dropdown(NEW_BOX_COLOR_OPTIONS, value=NEW_BOX_COLOR_OPTIONS[0], label="색상")
            new_box_number_input = gr.Dropdown(NEW_BOX_NUMBER_OPTIONS, value=NEW_BOX_NUMBER_OPTIONS[0], label="숫자")
            add_box_cancel_button = gr.Button("박스 추가 취소")
        table_output = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=TABLE_DATATYPES,
            row_count=0,
            column_count=(len(TABLE_HEADERS), "fixed"),
            interactive=True,
            type="array",
            label="탐지된 타일 목록 (편집 가능, 행 클릭 시 그 박스 하나만 표시)",
        )
        with gr.Row():
            filter_status_output = gr.Markdown()
            show_all_button = gr.Button("전체 보기")

        box_pending_state = gr.State(None)

        gr.Markdown(
            "### 5. 저장 (자른 사진 기준)\n"
            "저장해도 같은 이미지에 머무릅니다 — 다른 영역을 또 잘라서 계속 저장할 수 "
            "있습니다. 이 이미지가 끝났으면 위쪽 '다음 이미지' 버튼을 누르세요."
        )
        with gr.Row():
            split_input = gr.Radio(list(SPLITS), value="train", label="저장할 split")
            save_button = gr.Button("검수한 결과 저장", variant="primary")
        save_status_output = gr.Markdown()

        # --- folder navigation -------------------------------------------------
        load_folder_button.click(
            load_folder,
            inputs=[folder_path_input],
            outputs=[file_list_state, current_index_state, raw_image_input, folder_status_output],
        )
        next_button.click(
            go_to_next_image,
            inputs=[file_list_state, current_index_state],
            outputs=[folder_status_output, raw_image_input, file_list_state, current_index_state],
        )
        save_button.click(
            save_dataset,
            inputs=[working_state, table_output, split_input],
            outputs=[save_status_output],
        )

        # --- same crop/zoom/detect/review wiring as evaluate_ui.py -------------
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
        raw_image_input.change(
            clear_detection_outputs,
            inputs=[],
            outputs=[annotated_output, table_output, summary_output],
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
        zoom_button.click(
            apply_zoom,
            inputs=[cropped_state, zoom_slider],
            outputs=[working_state, zoomed_preview],
        )
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
        table_output.change(
            redraw_from_table,
            inputs=[working_state, table_output],
            outputs=[annotated_output],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Folder-batch review UI for Rummikub detections")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app(args.weights).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
    )
