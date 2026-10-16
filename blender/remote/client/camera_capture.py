"""Camera capture and crop helpers for the remote client."""
from __future__ import annotations
import sys
from typing import Optional, Union
import cv2


def backend_candidates(name: str) -> list[int]:
    if name == "dshow":
        return [cv2.CAP_DSHOW]
    if name == "msmf":
        return [cv2.CAP_MSMF]
    if name == "v4l2":
        return [cv2.CAP_V4L2]
    if name == "any":
        return [cv2.CAP_ANY]

    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def configure_capture(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    fps: float,
) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )


def open_camera(
    source: Union[int, str],
    backend: str,
    width: int,
    height: int,
    fps: float,
) -> tuple[cv2.VideoCapture, object]:
    failures: list[str] = []
    # File/URL sources must use a media backend, not a Windows webcam backend.
    backends = [cv2.CAP_ANY] if backend == "auto" and isinstance(source, str) else backend_candidates(backend)
    for api in backends:
        cap = cv2.VideoCapture(source, api)
        if not cap.isOpened():
            cap.release()
            failures.append(f"backend={api}: 打开失败")
            continue

        if isinstance(source, int):
            configure_capture(cap, width, height, fps)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, frame

        cap.release()
        failures.append(f"backend={api}: 已打开但读取不到图像")

    detail = "；".join(failures) if failures else "没有可用后端"
    raise RuntimeError(f"无法打开摄像头 {source!r}：{detail}")


def center_crop_square(frame, size: Optional[int] = None):
    height, width = frame.shape[:2]
    side = min(width, height) if size is None else size
    if side <= 0:
        raise ValueError("正方形裁剪边长必须大于 0")
    if side > min(width, height):
        raise ValueError(
            f"正方形裁剪边长 {side} 超过当前画面尺寸 {width}x{height}"
        )

    left = (width - side) // 2
    top = (height - side) // 2
    return frame[top : top + side, left : left + side]


def pad_square(
    frame,
    size: Optional[int] = None,
    border_value=(0, 0, 0),
):
    height, width = frame.shape[:2]
    side = max(width, height) if size is None else size
    if side <= 0:
        raise ValueError("正方形补边尺寸必须大于 0")
    if side < max(width, height):
        raise ValueError(
            f"正方形补边尺寸 {side} 小于当前画面尺寸 {width}x{height}"
        )

    left = (side - width) // 2
    right = side - width - left
    top = (side - height) // 2
    bottom = side - height - top
    return cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )
