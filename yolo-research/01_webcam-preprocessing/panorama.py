from datetime import datetime
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
CAMERA_INDEX = 1
MAX_CAMERA_PROBE = 4
OUTPUT_DIR = BASE_DIR / "panoramas"

WINDOW_NAME = "Panorama Capture"


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


def draw_overlay(frame, shot_count: int) -> None:
    cv2.putText(
        frame,
        f"캡처된 프레임: {shot_count}장  [SPACE] 캡처  [S] 파노라마 생성  [C] 카메라 변경  [Q] 종료",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def stitch_and_save(frames: list) -> Path | None:
    if len(frames) < 2:
        print("파노라마를 만들려면 최소 2장 이상 캡처해야 합니다.")
        return None

    stitcher = cv2.Stitcher_create()
    status, panorama = stitcher.stitch(frames)

    if status != cv2.Stitcher_OK:
        print(f"파노라마 생성 실패 (status={status})")
        return None

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"panorama_{timestamp}.jpg"
    cv2.imwrite(str(output_path), panorama)
    print(f"파노라마 저장 완료: {output_path}")
    return output_path


def main() -> None:
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

    frames = []

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("웹캠 프레임을 읽지 못했습니다.")
                break

            display_frame = frame.copy()
            draw_overlay(display_frame, len(frames))
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord(" "):
                frames.append(frame.copy())
                print(f"프레임 캡처됨 ({len(frames)}장)")

            elif key == ord("s"):
                stitch_and_save(frames)

            elif key == ord("c"):
                camera_pos = (camera_pos + 1) % len(available_cameras)
                cap.release()
                cap = open_camera(available_cameras[camera_pos])

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
