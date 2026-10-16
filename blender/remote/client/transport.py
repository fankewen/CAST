#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import queue
import socket
import time
from pathlib import Path
from typing import Union

from client_config import default_config_path, load_config as read_config
from motion_worker import CameraConfig, PendingFrame, motion_worker_process
from local_frame_stream import FRAME_HEADER, MAX_FRAME_BYTES, receive_exact


LOG = logging.getLogger("cast-transport")
DEFAULT_CONFIG = default_config_path()


def parse_camera_source(value) -> Union[int, str]:
    text = str(value)
    try:
        return int(text)
    except ValueError:
        return text


def load_config(path: Path) -> CameraConfig:
    data = read_config(path)
    return CameraConfig(
        server_url=str(data["server_url"]),
        api_key=str(data["api_key"]),
        fps=float(data.get("fps", 24)),
        camera=parse_camera_source(data.get("camera", 0)),
        capture_width=int(data.get("capture_width", 1280)),
        capture_height=int(data.get("capture_height", 720)),
        square_size=int(data.get("square_size", 720)),
        square_mode=str(data.get("square_mode", "crop")).lower(),
        jpeg_quality=int(data.get("jpeg_quality", 85)),
        request_timeout=float(data.get("request_timeout_seconds", 30)),
        pipeline_depth=int(data.get("pipeline_depth", 4)),
        rig=dict(data.get("rig") or {}),
        target_mode=str(data.get("target_mode", "human")),
        preview=False,
        preview_width=int(data.get("preview_width", 640)),
        preview_port=0,
        control_port=int(data.get("control_port", 5601)),
        pose_host=str(data.get("pose_host", "127.0.0.1")),
        pose_port=int(data.get("pose_port", 5599)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forward the persistent camera daemon stream to CAST.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--pose-host", default=None)
    parser.add_argument("--pose-port", type=int, default=None)
    parser.add_argument("--control-port", type=int, default=None)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="accepted for compatibility; preview runs in the separate daemon",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(args.config.resolve())
    if any(v is not None for v in (args.fps, args.pose_host, args.pose_port, args.control_port)):
        config = CameraConfig(
            **{
                **config.__dict__,
                "fps": float(args.fps) if args.fps is not None else config.fps,
                "pose_host": args.pose_host or config.pose_host,
                "pose_port": args.pose_port or config.pose_port,
                "control_port": args.control_port or config.control_port,
            }
        )

    data = json.loads(args.config.read_text(encoding="utf-8"))
    input_port = int(data.get("input_port", 5602))
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    frame_queue = context.Queue(maxsize=1)
    worker = context.Process(
        target=motion_worker_process,
        args=(config, frame_queue, stop_event, str(args.config.resolve())),
        name="cast-motion-worker",
        daemon=True,
    )
    worker.start()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", input_port))
    listener.listen(1)
    listener.settimeout(0.2)
    connection = None
    first_frame = True
    last_stats = time.monotonic()
    frame_count = 0

    try:
        while not stop_event.is_set():
            if not worker.is_alive():
                raise RuntimeError(
                    f"CAST motion worker exited unexpectedly with code {worker.exitcode}"
                )
            if connection is None:
                try:
                    connection, _address = listener.accept()
                    connection.settimeout(0.2)
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    LOG.info("STATUS CAMERA_FEED_CONNECTED port=%d", input_port)
                except socket.timeout:
                    continue
            header = receive_exact(connection, FRAME_HEADER.size, stop_event)
            if header is None:
                connection.close()
                connection = None
                continue
            timestamp_ms, payload_size = FRAME_HEADER.unpack(header)
            if not 0 < payload_size <= MAX_FRAME_BYTES:
                LOG.warning("STATUS INVALID_LOCAL_FRAME bytes=%d", payload_size)
                connection.close()
                connection = None
                continue
            payload = receive_exact(connection, payload_size, stop_event)
            if payload is None:
                connection.close()
                connection = None
                continue
            try:
                frame_queue.put_nowait(PendingFrame(payload, timestamp_ms))
            except queue.Full:
                # Retain the freshest image while the paced worker is waiting.
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait(PendingFrame(payload, timestamp_ms))
                except queue.Full:
                    pass
            frame_count += 1
            if first_frame:
                LOG.info("STATUS CAMERA_FEED_ACTIVE port=%d", input_port)
                first_frame = False
            now = time.monotonic()
            if now - last_stats >= 5.0:
                LOG.info(
                    "STATUS INPUT fps=%.1f",
                    frame_count / (now - last_stats),
                )
                frame_count = 0
                last_stats = now
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if connection is not None:
            connection.close()
        listener.close()
        worker.join(timeout=config.request_timeout + 2.0)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2.0)
        frame_queue.close()
        frame_queue.join_thread()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
