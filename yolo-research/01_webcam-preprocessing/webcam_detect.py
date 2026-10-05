from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_v8.pt"
CAMERA_INDEX = 1
MAX_CAMERA_PROBE = 4
CONFIDENCE = 0.25

WINDOW_NAME = "Rummikub YOLO Detection"
BUTTON_RECT = (10, 10, 170, 40)  # x, y, w, h

switch_requested = False


def on_mouse(event, x, y, _flags, _param) -> None:
    global switch_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = BUTTON_RECT
        if bx <= x <= bx + bw and by <= y <= by + bh:
            switch_requested = True


def find_available_cameras(max_index: int) -> list[int]:
    available = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
        cap.release()
    return available


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 {index}번을 열 수 없습니다.")
    return cap


def draw_button(frame, label: str) -> None:
    x, y, w, h = BUTTON_RECT
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(
        frame,
        label,
        (x + 10, y + h // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    global switch_requested

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH.resolve()}")

    device = 0 if torch.cuda.is_available() else "cpu"

    print("사용 장치:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(MODEL_PATH))

    available_cameras = find_available_cameras(MAX_CAMERA_PROBE)
    if not available_cameras:
        raise RuntimeError("사용 가능한 카메라를 찾지 못했습니다.")
    if CAMERA_INDEX in available_cameras:
        available_cameras.remove(CAMERA_INDEX)
        available_cameras.insert(0, CAMERA_INDEX)

    print("사용 가능한 카메라:", available_cameras)

    camera_pos = 0
    cap = open_camera(available_cameras[camera_pos])

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("웹캠 프레임을 읽지 못했습니다.")
                break

            results = model.predict(
                source=frame,
                conf=CONFIDENCE,
                device=device,
                verbose=False,
            )

            annotated_frame = results[0].plot()
            draw_button(
                annotated_frame,
                f"카메라 변경 (#{available_cameras[camera_pos]})",
            )

            cv2.imshow(WINDOW_NAME, annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("c") or switch_requested:
                switch_requested = False
                camera_pos = (camera_pos + 1) % len(available_cameras)
                cap.release()
                cap = open_camera(available_cameras[camera_pos])

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
