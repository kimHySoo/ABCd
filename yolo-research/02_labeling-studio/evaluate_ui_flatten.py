"""Folder-batch page that auto-flattens each image using green tape corners
before running the same crop -> zoom -> detect -> review -> save pipeline as
evaluate_ui.py. Saving advances to the next image automatically.

The flattening step (corner detection, corner ordering, homography, canonical
output size, post-warp tape removal) mirrors
`ABCD_AI/rummikub/app/services/calibration_service.py`
(`CalibrationService._detect_tape_corners` / `order_corners` / `calibrate` /
`warp`) and the extra tape-cleanup cell in
`ABCD_AI/rummikub/test/pipeline_crop_save.ipynb`, so images labeled here match
what the production pipeline actually feeds to YOLO."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

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

# Same green-tape HSV range as CalibrationService.GREEN_HSV_LOWER/UPPER.
DEFAULT_HUE_LOWER = 40
DEFAULT_HUE_UPPER = 85
TAPE_SAT_LOWER = 60
TAPE_VAL_LOWER = 60

# Same absolute-pixel-area cutoff as CalibrationService.MIN_TAPE_AREA_PX (no
# upper bound -- the service only takes the 4 largest surviving contours).
MIN_TAPE_AREA_PX = 200

# Same fixed real-world table size / scale as CalibrationService.TABLE_SIZE_CM
# and PX_PER_CM -- the warp target is a fixed canonical size, not derived from
# the measured corner distances (unlike the old blue-tape version).
DEFAULT_TABLE_WIDTH_CM = 110
DEFAULT_TABLE_HEIGHT_CM = 85
DEFAULT_PX_PER_CM = 10

# Same post-warp cleanup as the "테이프 제거" cell in pipeline_crop_save.ipynb.
TAPE_MASK_DILATE_PX = 5


# --- folder navigation (tracks the current file path, not pixel data) ------

def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _progress_text(file_list: list[str], index: int) -> str:
    if not file_list:
        return "불러온 이미지가 없습니다."
    if index >= len(file_list):
        return f"✅ 폴더의 이미지 {len(file_list)}장을 모두 처리했습니다."
    return f"{index + 1}/{len(file_list)}: {Path(file_list[index]).name}"


def _read_rgb_preview(path: str | None):
    """Read-only preview of the raw file, shown before any flattening runs."""
    if not path:
        return None
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_folder(folder_path: str):
    if not folder_path:
        return [], -1, None, None, None, "❌ 폴더 경로를 입력하세요."
    folder = Path(folder_path)
    if not folder.is_dir():
        return [], -1, None, None, None, f"❌ 폴더를 찾을 수 없습니다: {folder}"
    files = _list_images(folder)
    if not files:
        return [], -1, None, None, None, f"❌ 이미지 파일이 없습니다: {folder}"
    file_list = [str(p) for p in files]
    preview = _read_rgb_preview(file_list[0])
    status = f"✅ {len(files)}장을 불러왔습니다. {_progress_text(file_list, 0)} — 준비되면 '평탄화 실행'을 누르세요."
    return file_list, 0, file_list[0], preview, None, status


def advance_to_next(file_list: list[str], current_index: int):
    """Returns (file_list, index, path, raw_preview, cleared_flatten_result, progress_text).
    The 5th value is always None -- every navigation step clears whatever
    flattened image was showing, since it belonged to the previous file."""
    next_index = current_index + 1
    next_path = file_list[next_index] if next_index < len(file_list) else None
    preview = _read_rgb_preview(next_path)
    progress = _progress_text(file_list, next_index)
    if next_path:
        progress += " — 준비되면 '평탄화 실행'을 누르세요."
    return file_list, next_index, next_path, preview, None, progress


def go_to_next_image(file_list, index):
    """Moves on regardless of whether the current image was saved yet --
    save and "next" are independent actions so the same image can be
    cropped/detected/saved multiple times (once per region) before moving on."""
    file_list, next_index, next_path, preview, cleared_flatten, progress = advance_to_next(file_list, index)
    return f"다음 이미지로 이동했습니다. {progress}", file_list, next_index, next_path, preview, cleared_flatten


def clear_detection_outputs():
    return None, [], ""


# --- green-tape corner detection + perspective flatten (mirrors
# CalibrationService._detect_tape_corners / order_corners / calibrate / warp,
# plus the tape-removal cell from pipeline_crop_save.ipynb) -----------------

def order_corners(points: np.ndarray) -> np.ndarray:
    """Same sum/diff corner sort as CalibrationService.order_corners: the
    point with the smallest x+y is top-left, largest x+y is bottom-right,
    largest x-y is top-right, smallest x-y is bottom-left."""
    s = points.sum(axis=1)
    diff = points[:, 0] - points[:, 1]
    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmax(diff)]
    bottom_left = points[np.argmin(diff)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def detect_tape_corners(image_bgr: np.ndarray, hue_lower: int, hue_upper: int) -> np.ndarray:
    """Same detection as CalibrationService._detect_tape_corners: HSV inRange
    on the (green) tape color, open+close to clean the mask, keep contours
    above an absolute pixel-area floor, take the 4 largest, order them."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_lower, TAPE_SAT_LOWER, TAPE_VAL_LOWER])
    upper = np.array([hue_upper, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= MIN_TAPE_AREA_PX]
    if len(candidates) < 4:
        raise ValueError(f"초록 테이프 기준점을 4개 찾지 못했습니다 (검출: {len(candidates)}개)")
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)[:4]

    centers = []
    for contour in candidates:
        moments = cv2.moments(contour)
        centers.append([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    return order_corners(np.array(centers, dtype=np.float32))


def warp_canonical(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    table_width_cm: float,
    table_height_cm: float,
    px_per_cm: float,
) -> np.ndarray:
    """Same homography + fixed-size warp as CalibrationService.calibrate/warp:
    the output size comes from the configured real-world table size, not from
    the corners' measured pixel distance, and there is no orientation swap --
    `order_corners` already fixes top-left/top-right/etc regardless of camera
    rotation."""
    out_w = round(float(table_width_cm) * float(px_per_cm))
    out_h = round(float(table_height_cm) * float(px_per_cm))
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    homography, _ = cv2.findHomography(corners, dst)
    return cv2.warpPerspective(image_bgr, homography, (out_w, out_h))


def remove_tape(board_bgr: np.ndarray, hue_lower: int, hue_upper: int) -> np.ndarray:
    """Same cleanup as the "캘리브레이션 테이프 제거" cell in
    pipeline_crop_save.ipynb: after warping, the tape is still visible at the
    board edges and would otherwise get picked up as a foreground blob later,
    so it's masked out and filled with the board's median color."""
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_lower, TAPE_SAT_LOWER, TAPE_VAL_LOWER])
    upper = np.array([hue_upper, 255, 255])
    tape_mask = cv2.inRange(hsv, lower, upper)
    tape_mask = cv2.dilate(tape_mask, np.ones((TAPE_MASK_DILATE_PX, TAPE_MASK_DILATE_PX), np.uint8))

    bg_color = np.median(board_bgr.reshape(-1, 3), axis=0).astype(np.uint8)
    cleaned = board_bgr.copy()
    cleaned[tape_mask > 0] = bg_color
    return cleaned


def flatten_current(path, table_width_cm, table_height_cm, px_per_cm, hue_lower, hue_upper):
    if not path:
        return None, ""
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        return None, f"❌ 이미지를 읽을 수 없습니다: {Path(path).name}"
    hue_lower, hue_upper = int(hue_lower), int(hue_upper)
    try:
        corners = detect_tape_corners(image_bgr, hue_lower, hue_upper)
    except ValueError as exc:
        raw_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return raw_rgb, (
            f"⚠️ {Path(path).name}: {exc} — 원본을 그대로 불러왔습니다. "
            "Hue 범위를 조정하고 '평탄화 실행'을 다시 눌러보거나, 1-1번에서 직접 자르세요."
        )
    warped_bgr = warp_canonical(image_bgr, corners, table_width_cm, table_height_cm, px_per_cm)
    cleaned_bgr = remove_tape(warped_bgr, hue_lower, hue_upper)
    cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    height, width = cleaned_rgb.shape[:2]
    return cleaned_rgb, f"✅ {Path(path).name}: 테이프 인식 후 {width}x{height}로 평탄화(+테이프 제거) 완료."


def build_app(default_weights: Path = DEFAULT_WEIGHTS) -> gr.Blocks:
    with gr.Blocks(title="Rummikub 타일 탐지 검수 (폴더 + 평탄화)") as demo:
        gr.Markdown(
            "## 폴더 불러오기 → 평탄화(초록 테이프, ABCD_AI 프로덕션과 동일 방식) → "
            "(선택)자르기 → 확대 → 탐지 → 검수/수정 → 저장 → (준비되면) 다음 이미지\n"
            "이미지를 불러오는 것과 평탄화는 **분리된 별도 단계**입니다. 폴더를 불러오거나 "
            "다음 이미지로 넘어가도 평탄화는 자동으로 실행되지 않고, 원본 미리보기만 "
            "먼저 보여줍니다. 준비되면 '평탄화 실행'을 직접 눌러야 평탄화가 진행됩니다. "
            "**저장과 다음 이미지도 분리된 동작입니다** — 저장해도 자동으로 넘어가지 "
            "않으니, 같은 이미지에서 테이프 영역이 여러 개면 평탄화·자르기를 다시 해서 "
            "별도 샘플로 여러 번 저장한 뒤 준비되면 '다음 이미지'를 누르세요. 결과는 "
            f"매번 `{RUMMIKUB_DATA_ROOT}`에 저장됩니다."
        )

        file_list_state = gr.State([])
        current_index_state = gr.State(-1)
        current_path_state = gr.State(None)

        gr.Markdown("### 0. 폴더 불러오기")
        with gr.Row():
            folder_path_input = gr.Textbox(label="이미지 폴더 경로", placeholder=r"C:\path\to\images")
        with gr.Row():
            load_folder_button = gr.Button("폴더 불러오기", variant="primary")
            next_button = gr.Button("다음 이미지")
        folder_status_output = gr.Markdown()
        raw_loaded_preview = gr.Image(type="numpy", interactive=False, label="폴더에서 불러온 원본 이미지 (평탄화 전)")

        gr.Markdown(
            "### 1. 평탄화 (초록 테이프 기준, 버튼을 눌러야 실행됨)\n"
            "`ABCD_AI/rummikub`의 `CalibrationService`/`pipeline_crop_save.ipynb`와 "
            "동일한 방식입니다: 책상 네 모서리의 초록 테이프 4개를 찾아 원근 보정하고, "
            "실측 테이블 크기(cm) x 배율(px/cm)로 정해지는 고정 크기로 출력한 뒤 "
            "테이프 자국을 배경색으로 지웁니다. 테이프가 잘 안 잡히면 Hue 범위를 "
            "조정하고 '평탄화 실행'을 다시 누르세요."
        )
        with gr.Row():
            table_width_input = gr.Number(value=DEFAULT_TABLE_WIDTH_CM, label="테이블 가로 길이(cm)")
            table_height_input = gr.Number(value=DEFAULT_TABLE_HEIGHT_CM, label="테이블 세로 길이(cm)")
            px_per_cm_input = gr.Number(value=DEFAULT_PX_PER_CM, label="배율(px/cm)")
        with gr.Row():
            hue_lower_input = gr.Slider(0, 179, value=DEFAULT_HUE_LOWER, step=1, label="초록 테이프 Hue 하한")
            hue_upper_input = gr.Slider(0, 179, value=DEFAULT_HUE_UPPER, step=1, label="초록 테이프 Hue 상한")
        run_flatten_button = gr.Button("평탄화 실행", variant="primary")
        flatten_status_output = gr.Markdown()

        working_state = gr.State(None)  # image actually used for detect/save (crop, optionally zoomed)
        cropped_state = gr.State(None)  # crop result at 1x, kept so zoom can be re-applied from scratch
        raw_original_state = gr.State(None)  # clean copy of the flattened image, redrawn from on every click
        crop_pending_state = gr.State(None)
        crop_box_state = gr.State(None)

        gr.Markdown(
            "### 1-1. 추가로 자르기 (선택)\n"
            "평탄화 결과에서 더 좁게 자르고 싶을 때만 사용하세요. 오른쪽 미리보기를 "
            "클릭해서 자를 영역의 **첫 모서리**를 찍고 반대쪽 모서리를 다시 클릭한 뒤 "
            "'자르기 적용'을 누르세요."
        )
        with gr.Row():
            raw_image_input = gr.Image(type="numpy", interactive=False, label="평탄화된 이미지")
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
            "그 박스 하나만 표시됩니다. 겹치는 중복 박스는 자동으로 지워집니다. 클래스가 "
            "틀렸으면 셀을 더블클릭해서 고치세요.\n\n"
            "**놓친 타일이 있으면**: 아래에서 색상/숫자를 고른 뒤, 위 '탐지 결과' 이미지를 "
            "두 번 클릭하면 새 박스가 표에 추가됩니다. 색상은 다음 박스·다음 이미지에도 "
            "그대로 유지됩니다."
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
            "저장해도 같은 이미지에 머무릅니다 — 다른 영역을 다시 평탄화/자르기해서 "
            "계속 저장할 수 있습니다. 이 이미지가 끝났으면 위쪽 '다음 이미지' 버튼을 "
            "누르세요."
        )
        with gr.Row():
            split_input = gr.Radio(list(SPLITS), value="train", label="저장할 split")
            save_button = gr.Button("검수한 결과 저장", variant="primary")
        save_status_output = gr.Markdown()

        # --- folder navigation: loading/advancing/saving are deliberately NOT
        # chained into flattening, and saving is deliberately NOT chained into
        # advancing. They only show the raw preview and clear any previous
        # flattened result (raw_image_input=None) -- flattening itself only
        # ever runs when "평탄화 실행" is clicked, and "다음 이미지" is the only
        # thing that advances the folder position.
        load_folder_button.click(
            load_folder,
            inputs=[folder_path_input],
            outputs=[
                file_list_state,
                current_index_state,
                current_path_state,
                raw_loaded_preview,
                raw_image_input,
                folder_status_output,
            ],
        )
        next_button.click(
            go_to_next_image,
            inputs=[file_list_state, current_index_state],
            outputs=[
                folder_status_output,
                file_list_state,
                current_index_state,
                current_path_state,
                raw_loaded_preview,
                raw_image_input,
            ],
        )
        save_button.click(
            save_dataset,
            inputs=[working_state, table_output, split_input],
            outputs=[save_status_output],
        )
        run_flatten_button.click(
            flatten_current,
            inputs=[
                current_path_state,
                table_width_input,
                table_height_input,
                px_per_cm_input,
                hue_lower_input,
                hue_upper_input,
            ],
            outputs=[raw_image_input, flatten_status_output],
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
    parser = argparse.ArgumentParser(description="Folder-batch review UI with tape-corner flattening")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7863)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app(args.weights).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
    )
