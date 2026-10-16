#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import cv2

import lens as lens_mod
from camera_capture import center_crop_square, open_camera, pad_square
from local_frame_stream import FrameStreamSender
from client_config import default_config_path, load_config
from fps_control import FpsConfigWatcher


APP_DIR = Path(sys.executable if getattr(sys, "frozen", False) else __file__).parent
DEFAULT_CONFIG = default_config_path()


def load_settings(config_path: Path) -> dict:
    defaults = {
        "camera": 0,
        "fps": 24.0,
        "capture_width": 1280,
        "capture_height": 720,
        "square_size": 720,
        "square_mode": "crop",
        "jpeg_quality": 85,
        "preview_port": 5600,
        "input_port": 5602,
        # 镜头畸变矫正
        "lens_correct": False,
        "lens_strength": 1.0,
        "lens_profile": None,
    }
    if config_path.is_file():
        defaults.update(load_config(config_path))
    defaults["state_root"] = config_path.parent
    return defaults


def find_lens_profile(settings: dict):
    """按 配置指定 -> 项目目录 -> exe 目录 -> 打包内嵌 的顺序找标定档案。"""
    candidates = []
    explicit = settings.get("lens_profile")
    if explicit:
        candidates.append(settings["state_root"] / Path(explicit))
    candidates.append(settings["state_root"] / "camera_profile.json")
    for path in candidates:
        profile = lens_mod.load_profile(path)
        if profile is not None:
            return profile, str(path)
    return None, ""


def build_lens_maps(settings: dict, frame_size, log):
    """返回 (crop_maps, full_maps, 说明)。crop_maps 是"去畸变+中心裁切"一步到位。"""
    if not settings.get("lens_correct", True):
        return None, None, "disabled"
    profile, where = find_lens_profile(settings)
    if profile is None:
        return None, None, "profile not found"
    prof_cam = profile["raw"].get("camera")
    try:
        cam_int = int(settings.get("camera"))
    except (TypeError, ValueError):
        cam_int = None
    if prof_cam is not None and cam_int is not None and int(prof_cam) != cam_int:
        return None, None, f"profile is for camera {prof_cam}, configured camera is {cam_int}"

    strength = float(settings.get("lens_strength", 1.0))
    crop_size = int(settings["square_size"])
    k1 = float(profile["dist_coeffs"].reshape(-1)[0]) * strength
    note = f"k1={k1:+.3f} strength={strength:g} profile={where}"

    if settings.get("square_mode", "crop") == "pad":
        full = lens_mod.full_maps_for(profile, frame_size, strength=strength)
        return None, full, note
    mx, my, zoom, bad = lens_mod.maps_for(profile, frame_size, crop_size, strength=strength)
    note += f" zoom={zoom:.3f} out_of_range={bad * 100:.4f}%"
    return (mx, my), None, note


class PreviewWindow:
    def __init__(self, title: str, max_width: int) -> None:
        # max_width sets only the initial size; later sizes belong to the user.
        self.initial_width = max(160, max_width)
        self.initialized = False
        self.closed = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(True, True)
        self.root.minsize(160, 160)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.canvas = tk.Canvas(self.root, background="black", highlightthickness=0,
                                borderwidth=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.CENTER)
        self.photo = None

    def show(self, frame) -> bool:
        if self.closed:
            return False
        frame_height, frame_width = frame.shape[:2]
        if not self.initialized:
            width = min(frame_width, self.initial_width)
            height = max(160, round(frame_height * width / frame_width))
            self.root.geometry(f"{width}x{height}")
            self.initialized = True
        # Process border drags and close events before accessing the canvas.
        self.root.update()
        if self.closed:
            return False
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / frame_width, height / frame_height)
        display_size = (max(1, round(frame_width * scale)),
                        max(1, round(frame_height * scale)))
        if display_size != (frame_width, frame_height):
            frame = cv2.resize(
                frame,
                display_size,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width = rgb.shape[:2]
        ppm = f"P6\n{image_width} {image_height}\n255\n".encode("ascii") + rgb.tobytes()
        self.photo = tk.PhotoImage(master=self.root, data=ppm, format="PPM")
        self.canvas.coords(self.image_item, width / 2, height / 2)
        self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.root.update_idletasks()
        return not self.closed

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def parse_camera_source(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Camera capture, optional preview and local TCP frame transport.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--square-size", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=720)
    parser.add_argument("--lens-correct", dest="lens_correct", action="store_true", default=None,
                        help="开启镜头畸变矫正")
    parser.add_argument("--no-lens-correct", dest="lens_correct", action="store_false", default=None,
                        help="关闭镜头畸变矫正")
    parser.add_argument("--lens-strength", type=float, default=None, help="矫正强度，默认 1.0")
    parser.add_argument("--lens-profile", type=Path, default=None, help="标定档案路径")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings(args.config.resolve())
    if args.camera is not None:
        settings["camera"] = args.camera
    if args.fps is not None:
        settings["fps"] = args.fps
    if args.width is not None:
        settings["capture_width"] = args.width
    if args.height is not None:
        settings["capture_height"] = args.height
    if args.square_size is not None:
        settings["square_size"] = args.square_size
    if args.lens_correct is not None:
        settings["lens_correct"] = args.lens_correct
    if args.lens_strength is not None:
        settings["lens_strength"] = args.lens_strength
    if args.lens_profile is not None:
        settings["lens_profile"] = args.lens_profile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.config.resolve().parent / "camera_daemon.log", mode="w", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("camera-standalone")

    transport = FrameStreamSender("127.0.0.1", int(settings["input_port"]))
    cap = None
    window = PreviewWindow("CAST Camera", args.max_width) if args.preview else None
    intervals = deque(maxlen=120)
    previous_capture = None
    last_stats = time.monotonic()
    frame_count = 0
    last_transport_warning = 0.0

    try:
        cap, frame = open_camera(
            parse_camera_source(settings["camera"]),
            "auto",
            int(settings["capture_width"]),
            int(settings["capture_height"]),
            float(settings["fps"]),
        )
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(
            "STATUS CAMERA_READY requested=%dx%d@%.2f actual=%dx%d crop=%s:%d",
            int(settings["capture_width"]),
            int(settings["capture_height"]),
            float(settings["fps"]),
            actual_width,
            actual_height,
            settings["square_mode"],
            int(settings["square_size"]),
        )

        lens_crop, lens_full, lens_note = build_lens_maps(
            settings, (actual_width, actual_height), log
        )
        if lens_crop is None and lens_full is None:
            log.info("STATUS LENS_OFF reason=%s", lens_note)
        else:
            log.info("STATUS LENS_ON %s", lens_note)

        period = 1.0 / float(settings["fps"])
        fps_watch = FpsConfigWatcher(args.config.resolve(), settings["fps"])
        next_deadline = time.monotonic()

        while window is None or not window.closed:
            target_fps = fps_watch.poll()
            if target_fps != float(settings["fps"]):
                settings["fps"] = target_fps
                cap.set(cv2.CAP_PROP_FPS, target_fps)
                period = 1.0 / target_fps
                next_deadline = time.monotonic()
                intervals.clear()
                log.info("STATUS INPUT_TARGET fps=%.1f", target_fps)
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("STATUS CAMERA_RETRY")
                time.sleep(0.05)
                next_deadline = time.monotonic() + period
                continue

            captured_at = time.monotonic()
            if previous_capture is not None:
                intervals.append(captured_at - previous_capture)
            previous_capture = captured_at

            if lens_crop is not None:
                # 去畸变 + 中心裁切一步完成
                frame = cv2.remap(frame, lens_crop[0], lens_crop[1], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            else:
                if lens_full is not None:
                    frame = cv2.remap(frame, lens_full[0], lens_full[1], cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
                if settings["square_mode"] == "pad":
                    frame = pad_square(frame, int(settings["square_size"]))
                else:
                    frame = center_crop_square(frame, int(settings["square_size"]))

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(settings["jpeg_quality"])],
            )
            if not ok:
                continue

            payload = encoded.tobytes()
            timestamp_ms = time.time() * 1000.0
            try:
                delivered = transport.send(timestamp_ms, payload)
            except ValueError:
                log.exception("STATUS LOCAL_TRANSPORT_ERROR")
                delivered = False
            if not delivered and time.monotonic() - last_transport_warning >= 5.0:
                log.warning("STATUS LOCAL_TRANSPORT_WAITING port=%d", int(settings["input_port"]))
                last_transport_warning = time.monotonic()

            if window is not None and not window.show(frame):
                break

            frame_count += 1
            now = time.monotonic()
            if now - last_stats >= 5.0:
                average = frame_count / (now - last_stats)
                if intervals:
                    ordered = sorted(intervals)
                    p95 = ordered[int(len(ordered) * 0.95) - 1]
                    maximum = ordered[-1]
                else:
                    p95 = maximum = 0.0
                log.info(
                    "STATUS CAPTURE fps=%.1f p95_interval_ms=%.1f max_interval_ms=%.1f",
                    average,
                    p95 * 1000.0,
                    maximum * 1000.0,
                )
                frame_count = 0
                last_stats = now

            next_deadline += period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period:
                next_deadline = time.monotonic()
    finally:
        transport.close()
        if cap is not None:
            cap.release()
        if window is not None:
            window.close()
        log.info("STATUS CAMERA_STOPPED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
