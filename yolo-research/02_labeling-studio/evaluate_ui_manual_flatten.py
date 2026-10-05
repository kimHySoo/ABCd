"""Folder-batch page that flattens each image using MANUALLY picked corners
(you click the table's 4 corners yourself) before running the same
crop -> zoom -> detect -> review -> save pipeline as evaluate_ui.py. Saving
advances to the next image automatically.

The corner-picking + warp logic mirrors the `QuadPicker` / `order_quad` /
`flatten_table(quad=...)` manual-quad path in
`ABCD_AI/rummikub/test/test0729/test.ipynb`: click the 4 corners in order
(top-left -> top-right -> bottom-right -> bottom-left), the points get
re-ordered defensively the same way the notebook does, and the result is
warped to a fixed output size (850x550 by default, matching the notebook's
`BOARD_OUTPUT_SIZE`). Use this page when the automatic tape-corner detection
in `evaluate_ui_flatten.py` gets the corners wrong (tile clutter, no tape,
tape color out of range, etc.)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from evaluate_ui import (
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

# ABCD_AI's tile-detector checkpoint, not yolo_boardgame's local best.pt --
# this page's images are meant to match what that model was trained on.
DEFAULT_WEIGHTS = Path(
    r"C:\Users\SSAFY\workspace\ABCD_AI\rummikub\weights\tile_detector\best_yolo_2.pt"
)

# Same fixed output size as the notebook's BOARD_OUTPUT_SIZE = (850, 550).
DEFAULT_BOARD_WIDTH_PX = 850
DEFAULT_BOARD_HEIGHT_PX = 550

# Same click order as QuadPicker.LABELS in test.ipynb.
CORNER_LABELS = ["좌상(TL)", "우상(TR)", "우하(BR)", "좌하(BL)"]


# --- folder navigation (tracks the current file path, not pixel data) ------

def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _read_rgb(path: str | None):
    if not path:
        return None
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _progress_text(file_list: list[str], index: int) -> str:
    if not file_list:
        return "불러온 이미지가 없습니다."
    if index >= len(file_list):
        return f"✅ 폴더의 이미지 {len(file_list)}장을 모두 처리했습니다."
    return f"{index + 1}/{len(file_list)}: {Path(file_list[index]).name}"


def load_folder(folder_path: str):
    if not folder_path:
        return [], -1, None, None, None, [], None, "❌ 폴더 경로를 입력하세요."
    folder = Path(folder_path)
    if not folder.is_dir():
        return [], -1, None, None, None, [], None, f"❌ 폴더를 찾을 수 없습니다: {folder}"
    files = _list_images(folder)
    if not files:
        return [], -1, None, None, None, [], None, f"❌ 이미지 파일이 없습니다: {folder}"
    file_list = [str(p) for p in files]
    image = _read_rgb(file_list[0])
    status = (
        f"✅ {len(files)}장을 불러왔습니다. {_progress_text(file_list, 0)} — "
        f"이미지 위를 {' → '.join(CORNER_LABELS)} 순서로 4번 클릭하세요."
    )
    return file_list, 0, file_list[0], image, image, [], None, status


def advance_to_next(file_list: list[str], current_index: int):
    """Returns (file_list, index, path, raw_state, corner_preview,
    cleared_points, cleared_flatten, progress_text). Every navigation step
    clears the previous image's corner picks and flattened result -- they
    belonged to the image that was just left."""
    next_index = current_index + 1
    next_path = file_list[next_index] if next_index < len(file_list) else None
    image = _read_rgb(next_path)
    progress = _progress_text(file_list, next_index)
    if next_path:
        progress += f" — 이미지 위를 {' → '.join(CORNER_LABELS)} 순서로 4번 클릭하세요."
    return file_list, next_index, next_path, image, image, [], None, progress


def go_to_next_image(file_list, index):
    """Moves on regardless of whether the current image was saved yet --
    save and "next" are independent actions so the same image can be
    picked/cropped/detected/saved multiple times (once per region) before
    moving on."""
    file_list, next_index, next_path, raw_state, corner_preview, points, cleared_flatten, progress = advance_to_next(
        file_list, index
    )
    return (
        f"다음 이미지로 이동했습니다. {progress}",
        file_list,
        next_index,
        next_path,
        raw_state,
        corner_preview,
        points,
        cleared_flatten,
    )


def clear_detection_outputs():
    return None, [], ""


# --- manual corner picking + perspective flatten (mirrors QuadPicker /
# order_quad / flatten_table(quad=...) in test0729/test.ipynb) -------------

def order_quad(points: np.ndarray) -> np.ndarray:
    """Same defensive re-ordering as the notebook's order_quad: sort by angle
    around the centroid, rotate so the smallest x+y point comes first (top
    -left), then flip to clockwise order if the cross product says the click
    order came out counter-clockwise. Lets clicks be slightly out of order or
    the shot be rotated without producing a self-intersecting quad."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = int(np.argmin(points.sum(axis=1)))
    points = np.roll(points, -start, axis=0)
    if np.cross(points[1] - points[0], points[2] - points[1]) < 0:
        points = points[[0, 3, 2, 1]]
    return points.astype(np.float32)


def _draw_quad_preview(image: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    preview = np.asarray(image).copy()
    for i, (x, y) in enumerate(points):
        center = (int(round(x)), int(round(y)))
        cv2.drawMarker(preview, center, (0, 255, 255), cv2.MARKER_CROSS, 20, 3)
        cv2.putText(
            preview, str(i + 1), (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA,
        )
    if len(points) >= 2:
        chain = list(points) + ([points[0]] if len(points) == 4 else [])
        for (ax, ay), (bx, by) in zip(chain, chain[1:]):
            cv2.line(preview, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 255), 2)
    return preview


def _corner_status(points: list[tuple[float, float]]) -> str:
    if len(points) < 4:
        return f"{CORNER_LABELS[len(points)]} 점을 클릭하세요 ({len(points)}/4)"
    return "✅ 4점 선택 완료. '평탄화 적용'을 누르세요."


def handle_corner_click(raw_image, points, evt: gr.SelectData):
    if raw_image is None:
        return None, points or [], "먼저 이미지를 불러오세요."
    points = list(points or [])
    if len(points) >= 4:
        return _draw_quad_preview(raw_image, points), points, "이미 4점을 선택했습니다. 다시 찍으려면 '모서리 초기화'를 누르세요."
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        return _draw_quad_preview(raw_image, points), points, "좌표를 읽지 못했습니다. 다시 클릭하세요."
    points.append((float(index[0]), float(index[1])))
    return _draw_quad_preview(raw_image, points), points, _corner_status(points)


def undo_last_corner(raw_image, points):
    if raw_image is None:
        return None, [], "먼저 이미지를 불러오세요."
    points = list(points or [])
    if points:
        points.pop()
    return _draw_quad_preview(raw_image, points), points, _corner_status(points)


def reset_corners(raw_image, points):
    if raw_image is None:
        return None, [], "먼저 이미지를 불러오세요."
    return np.asarray(raw_image).copy(), [], _corner_status([])


def apply_manual_flatten(raw_image, points, board_width, board_height):
    if raw_image is None:
        return None, "❌ 먼저 이미지를 불러오세요."
    points = list(points or [])
    if len(points) != 4:
        return None, (
            f"❌ 4개의 모서리를 {' → '.join(CORNER_LABELS)} 순서로 클릭하세요 (현재 {len(points)}개)."
        )
    quad = order_quad(np.array(points, dtype=np.float32))
    width, height = int(board_width), int(board_height)
    if width < 2 or height < 2:
        return None, "❌ 출력 크기가 너무 작습니다."
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(quad, destination)
    flattened = cv2.warpPerspective(np.asarray(raw_image), matrix, (width, height), flags=cv2.INTER_CUBIC)
    return flattened, f"✅ {width}x{height}로 평탄화 완료. 필요하면 1-1에서 추가로 자르세요."


def build_app(default_weights: Path = DEFAULT_WEIGHTS) -> gr.Blocks:
    with gr.Blocks(title="Rummikub 타일 탐지 검수 (폴더 + 수동 모서리 평탄화)") as demo:
        gr.Markdown(
            "## 폴더 불러오기 → 모서리 직접 클릭 → 평탄화 → (선택)자르기 → 확대 → 탐지 → "
            "검수/수정 → 저장 → (준비되면) 다음 이미지\n"
            "`ABCD_AI/rummikub/test/test0729/test.ipynb`의 수동 모서리 선택(`QuadPicker`) "
            "방식과 동일합니다 — 초록 테이프 자동 검출이 실패하거나(타일에 가려짐, 테이프 "
            "없음 등) 잘못된 모서리를 잡을 때 이 페이지에서 직접 4점을 클릭해 평탄화하세요. "
            "이미지를 불러오는 것과 평탄화는 **분리된 별도 단계**입니다. 폴더를 불러오거나 "
            "다음 이미지로 넘어가도 평탄화는 자동으로 실행되지 않습니다. **저장과 다음 "
            "이미지도 분리된 동작입니다** — 저장해도 자동으로 넘어가지 않으니, 같은 "
            "이미지에서 영역이 여러 개면 모서리 선택·평탄화·자르기를 다시 해서 별도 샘플로 "
            f"여러 번 저장한 뒤 준비되면 '다음 이미지'를 누르세요. 결과는 매번 "
            f"`{RUMMIKUB_DATA_ROOT}`에 저장됩니다."
        )

        file_list_state = gr.State([])
        current_index_state = gr.State(-1)
        current_path_state = gr.State(None)
        loaded_raw_state = gr.State(None)  # clean copy of the current file's pixels, redrawn from on every click
        corner_points_state = gr.State([])  # clicked (x, y) points so far, in click order

        gr.Markdown("### 0. 폴더 불러오기")
        with gr.Row():
            folder_path_input = gr.Textbox(label="이미지 폴더 경로", placeholder=r"C:\path\to\images")
        with gr.Row():
            load_folder_button = gr.Button("폴더 불러오기", variant="primary")
            next_button = gr.Button("다음 이미지")
        folder_status_output = gr.Markdown()

        gr.Markdown(
            "### 1. 모서리 직접 클릭 (좌상 → 우상 → 우하 → 좌하 순서)\n"
            "아래 이미지를 테이블의 네 모서리 순서대로 4번 클릭하세요. 잘못 찍었으면 "
            "'마지막 점 취소'로 하나씩 되돌리거나 '모서리 초기화'로 전부 지울 수 있습니다."
        )
        corner_click_preview = gr.Image(
            type="numpy", interactive=True, label="여기를 클릭해서 모서리 4개 선택"
        )
        with gr.Row():
            undo_corner_button = gr.Button("마지막 점 취소")
            reset_corner_button = gr.Button("모서리 초기화")
        with gr.Row():
            board_width_input = gr.Number(value=DEFAULT_BOARD_WIDTH_PX, label="평탄화 출력 가로(px)")
            board_height_input = gr.Number(value=DEFAULT_BOARD_HEIGHT_PX, label="평탄화 출력 세로(px)")
        run_flatten_button = gr.Button("평탄화 적용", variant="primary")
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
            "저장해도 같은 이미지에 머무릅니다 — 다른 영역을 다시 모서리 선택/평탄화/자르기해서 "
            "계속 저장할 수 있습니다. 이 이미지가 끝났으면 위쪽 '다음 이미지' 버튼을 "
            "누르세요."
        )
        with gr.Row():
            split_input = gr.Radio(list(SPLITS), value="train", label="저장할 split")
            save_button = gr.Button("검수한 결과 저장", variant="primary")
        save_status_output = gr.Markdown()

        # --- folder navigation --------------------------------------------
        load_folder_button.click(
            load_folder,
            inputs=[folder_path_input],
            outputs=[
                file_list_state,
                current_index_state,
                current_path_state,
                loaded_raw_state,
                corner_click_preview,
                corner_points_state,
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
                loaded_raw_state,
                corner_click_preview,
                corner_points_state,
                raw_image_input,
            ],
        )
        save_button.click(
            save_dataset,
            inputs=[working_state, table_output, split_input],
            outputs=[save_status_output],
        )

        # --- manual corner picking ------------------------------------------
        corner_click_preview.select(
            handle_corner_click,
            inputs=[loaded_raw_state, corner_points_state],
            outputs=[corner_click_preview, corner_points_state, flatten_status_output],
        )
        undo_corner_button.click(
            undo_last_corner,
            inputs=[loaded_raw_state, corner_points_state],
            outputs=[corner_click_preview, corner_points_state, flatten_status_output],
        )
        reset_corner_button.click(
            reset_corners,
            inputs=[loaded_raw_state, corner_points_state],
            outputs=[corner_click_preview, corner_points_state, flatten_status_output],
        )
        run_flatten_button.click(
            apply_manual_flatten,
            inputs=[loaded_raw_state, corner_points_state, board_width_input, board_height_input],
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
    parser = argparse.ArgumentParser(description="Folder-batch review UI with manual quad-corner flattening")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7866)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app(args.weights).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
    )
