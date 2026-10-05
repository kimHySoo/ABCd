"""Minimal standalone page: upload an image and adjust its brightness.
No YOLO/detection involved -- just a live brightness preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np


def adjust_brightness(image, brightness):
    if image is None:
        return None
    brightness = int(brightness)
    if brightness == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="이미지 밝기 조정") as demo:
        gr.Markdown(
            "## 이미지 밝기 조정\n"
            "이미지를 업로드하고 슬라이더로 밝기를 조정하세요. 색상(Hue)은 그대로 두고 "
            "명도(HSV의 V값)만 올리거나 내려서, 단순히 RGB 값에 더하는 방식보다 색이 "
            "덜 틀어집니다. 결과 이미지 우측 상단 아이콘으로 다운로드할 수 있습니다."
        )
        with gr.Row():
            image_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="원본 이미지")
            image_output = gr.Image(type="numpy", interactive=False, label="조정된 이미지")
        brightness_slider = gr.Slider(-100, 100, value=0, step=1, label="밝기")

        image_input.change(
            adjust_brightness,
            inputs=[image_input, brightness_slider],
            outputs=[image_output],
        )
        brightness_slider.change(
            adjust_brightness,
            inputs=[image_input, brightness_slider],
            outputs=[image_output],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone image brightness adjustment UI")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7864)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
    )
