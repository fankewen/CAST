"""Ordered remote inference and pose forwarding in an external Python process."""
from __future__ import annotations
import json
import logging
import queue
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from client_config import write_json
from fps_control import FpsConfigWatcher, SubmissionPacer
from pipeline_client import PipelinedMotionClient

LOG = logging.getLogger("cast-remote-worker")
POSE_HEADER = struct.Struct("<II")

@dataclass(frozen=True)
class CameraConfig:
    server_url: str
    api_key: str
    fps: float
    camera: Union[int, str]
    capture_width: int
    capture_height: int
    square_size: int
    square_mode: str
    jpeg_quality: int
    request_timeout: float
    pipeline_depth: int
    rig: dict
    target_mode: str
    preview: bool
    preview_width: int
    preview_port: int
    control_port: int
    pose_host: str
    pose_port: int


@dataclass(frozen=True)
class PendingFrame:
    jpeg: bytes
    timestamp_ms: float


class PoseSender:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.socket: Optional[socket.socket] = None

    def _connect(self) -> bool:
        try:
            self.socket = socket.create_connection(
                (self.host, self.port),
                timeout=0.1,
            )
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            LOG.info("STATUS BLENDER_CONNECTED %s:%d", self.host, self.port)
            return True
        except OSError:
            self.socket = None
            return False

    def send(self, rot6d, frame_index: int) -> bool:
        rows = [[float(value) for value in row] for row in rot6d]
        if not rows or any(len(row) != 6 for row in rows):
            raise ValueError("rot6d 必须是 [J,6] 数组")
        values = [value for row in rows for value in row]
        payload = (
            POSE_HEADER.pack(frame_index & 0xFFFFFFFF, len(rows))
            + struct.pack(f"<{len(values)}f", *values)
        )

        for _ in range(1):
            if self.socket is None and not self._connect():
                continue
            try:
                self.socket.sendall(payload)
                return True
            except OSError:
                self.close()
        return False

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None


def motion_worker_process(
    config: CameraConfig,
    frame_queue,
    stop_event,
    fps_config_path=None,
) -> None:
    state_root = Path(fps_config_path).resolve().parent
    active_session_path = state_root/'active_session.json'
    file_handler = logging.FileHandler(state_root/'worker.log', mode="w", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), file_handler],
    )
    client = PipelinedMotionClient(
        config.server_url,
        config.api_key,
        window=config.pipeline_depth,
        timeout=config.request_timeout,
    )
    pose_sender = PoseSender(config.pose_host, config.pose_port)
    reset_event = threading.Event()
    paused_event = threading.Event()
    recover_event = threading.Event()
    rig_data = dict(config.rig or {})
    if not rig_data and (state_root/"active_rig.json").is_file():
        try:
            rig_data = json.loads(
                (state_root/"active_rig.json").read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            rig_data = {}

    def drain_frames() -> None:
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                return

    def control_server() -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", config.control_port))
        listener.listen(1)
        listener.settimeout(0.2)
        try:
            while not stop_event.is_set():
                try:
                    connection, _address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with connection:
                    try:
                        raw = connection.recv(4096)
                        request = json.loads(raw.decode("utf-8"))
                        command = str(request.get("cmd", "")).lower()
                        if command == "reset":
                            reset_event.set()
                            reply = {"ok": True, "command": "reset"}
                        elif command == "pause":
                            paused_event.set()
                            LOG.info("STATUS PAUSED")
                            reply = {"ok": True, "command": "pause"}
                        elif command == "stop":
                            stop_event.set()
                            LOG.info("STATUS DISCONNECTED")
                            reply = {"ok": True, "command": "stop"}
                        elif command == "resume":
                            paused_event.clear()
                            LOG.info("STATUS RESUMED")
                            reply = {"ok": True, "command": "resume"}
                        elif command == "disconnect":
                            stop_event.set()
                            LOG.info("STATUS DISCONNECTED")
                            reply = {"ok": True, "command": "disconnect"}
                        else:
                            reply = {"ok": False, "error": "unknown command"}
                    except Exception as exc:
                        reply = {"ok": False, "error": str(exc)}
                    connection.sendall(
                        (json.dumps(reply) + "\n").encode("utf-8")
                    )
        finally:
            listener.close()

    control_thread = threading.Thread(target=control_server, daemon=True)
    control_thread.start()
    retry_delay = 0.5
    reset_confirmation_pending = False
    confirmed_frames = 0
    using_imported_rig = False
    pending_futures = deque()
    fps_watch = FpsConfigWatcher(fps_config_path, config.fps)
    input_pacer = SubmissionPacer(config.fps)
    submit_stats_started = time.perf_counter()
    submitted_count = 0

    def start_fresh_session():
        fresh = PipelinedMotionClient(
            config.server_url,
            config.api_key,
            window=config.pipeline_depth,
            timeout=config.request_timeout,
        )
        try:
            session = fresh.start(rig_data, target_mode=config.target_mode)
        except BaseException:
            fresh.close()
            raise
        active = {
            "session_id": session["session_id"],
            "epoch": session["epoch"],
            "pipeline_capacity": session.get("pipeline_capacity"),
            "recommended_inflight": session.get("recommended_inflight"),
            "protocol_version": session.get("protocol_version"),
            "joints": session.get("joints"),
            "joint_names": session.get("joint_names", []),
        }
        active["server_url"] = config.server_url
        active["target_mode"] = config.target_mode
        write_json(active_session_path, active)
        return fresh, session

    try:
        while not stop_event.is_set():
            if recover_event.is_set():
                LOG.info("STATUS RECOVERING")
                for future in list(pending_futures):
                    try:
                        future.result()
                    except Exception:
                        pass
                pending_futures.clear()
                try:
                    client.close()
                except Exception:
                    pass
                active_session_path.unlink(missing_ok=True)
                drain_frames()

                recover_delay = 0.5
                while not stop_event.is_set():
                    try:
                        client, session = start_fresh_session()
                        using_imported_rig = bool(rig_data)
                        reset_confirmation_pending = False
                        confirmed_frames = 0
                        LOG.info(
                            "STATUS RECONNECTED session=%s epoch=%s capacity=%s",
                            session.get("session_id"),
                            session.get("epoch"),
                            session.get("pipeline_capacity"),
                        )
                        break
                    except Exception as exc:
                        LOG.warning(
                            "STATUS RECOVERY_RETRY delay=%.1f error=%s",
                            recover_delay,
                            exc,
                        )
                        stop_event.wait(recover_delay)
                        recover_delay = min(recover_delay * 2.0, 10.0)
                recover_event.clear()
                continue

            if reset_event.is_set() and not pending_futures:
                try:
                    client.reset()
                    drain_frames()
                    LOG.info("STATUS CACHE_RESET")
                    reset_confirmation_pending = True
                except Exception as exc:
                    LOG.warning("ERROR RESET_FAILED %s", exc)
                finally:
                    reset_event.clear()

            if client.sid is None and active_session_path.is_file():
                try:
                    meta = json.loads(
                        active_session_path.read_text(encoding="utf-8")
                    )
                    if meta.get("server_url") != config.server_url or meta.get("target_mode") != config.target_mode:
                        raise RuntimeError("Connection changed; import the FBX again")
                    client.adopt_session(
                        meta["session_id"],
                        meta["epoch"],
                        meta.get("pipeline_capacity"),
                    )
                    using_imported_rig = True
                    LOG.info(
                        "STATUS SESSION_ADOPTED session=%s joints=%s",
                        meta["session_id"],
                        meta.get("joints"),
                    )
                except Exception as exc:
                    active_session_path.unlink(missing_ok=True)
                    LOG.warning("STATUS SESSION_ADOPT_FAILED %s", exc)

            if client.sid is None and using_imported_rig:
                LOG.error("STATUS REIMPORT_REQUIRED")
                stop_event.wait(1.0)
                continue

            if client.sid is None:
                try:
                    session = client.start(rig_data, target_mode=config.target_mode)
                    LOG.info(
                        "STATUS CONNECTED session=%s epoch=%s capacity=%s joints=%d",
                        session.get("session_id"),
                        session.get("epoch"),
                        session.get("pipeline_capacity"),
                        int(session.get("joints", 0)),
                    )
                    retry_delay = 0.5
                except Exception as exc:
                    LOG.warning(
                        "STATUS RETRY_CONNECT delay=%.1f error=%s",
                        retry_delay,
                        exc,
                    )
                    stop_event.wait(retry_delay)
                    retry_delay = min(retry_delay * 2.0, 10.0)
                    continue

            # Apply completed responses in submission order.
            while pending_futures and pending_futures[0].done():
                future = pending_futures.popleft()
                try:
                    result = future.result()
                except Exception as exc:
                    LOG.warning("STATUS FRAME_ERROR %s", exc)
                    active_session_path.unlink(missing_ok=True)
                    recover_event.set()
                    break
                rot6d = result.get("motion", {}).get("rot6d")
                if rot6d and result.get("motion_valid", False):
                    pose_sender.send(rot6d, int(result.get("seq", 0)))
                confirmed_frames += 1
                if reset_confirmation_pending:
                    LOG.info(
                        "STATUS RESET_CONFIRMED seq=%s",
                        result.get("seq"),
                    )
                    reset_confirmation_pending = False
                elif confirmed_frames % 300 == 0:
                    LOG.info(
                        "STATUS STREAMING seq=%s",
                        result.get("seq"),
                    )

            # A failed future marks the HTTP client failed. Do not submit one
            # more frame and then leave the worker loop; let the recovery
            # branch at the top of the next iteration drain and reconnect.
            if recover_event.is_set():
                continue

            # A later request may fail before the ordered head completes.
            # Detect it now so we stop feeding a client already marked failed.
            for future in pending_futures:
                if future.done() and future.exception() is not None:
                    LOG.warning("STATUS FRAME_ERROR %s", future.exception())
                    active_session_path.unlink(missing_ok=True)
                    recover_event.set()
                    break
            if recover_event.is_set():
                continue

            if stop_event.is_set():
                break

            if paused_event.is_set() or reset_event.is_set():
                drain_frames()
                stop_event.wait(0.002)
                continue

            now = time.perf_counter()
            input_pacer.update(fps_watch.poll(now))
            delay = input_pacer.delay(now)
            if delay > 0:
                # Windows Event.wait rounds short waits to the system tick
                # (~15.6 ms); Python's high-resolution sleep keeps 24 Hz pacing.
                time.sleep(min(delay, 0.005))
                continue
            try:
                pending_frame = frame_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.002)
                continue

            try:
                future = client.submit(
                    pending_frame.jpeg,
                    pending_frame.timestamp_ms,
                )
            except BufferError:
                # Window full: keep only the newest unsubmitted frame.
                continue
            except Exception as exc:
                LOG.warning("STATUS FRAME_ERROR %s", exc)
                active_session_path.unlink(missing_ok=True)
                recover_event.set()
                continue
            pending_futures.append(future)
            now = time.perf_counter()
            input_pacer.submitted(now)
            submitted_count += 1
            if now - submit_stats_started >= 5.0:
                LOG.info("STATUS SUBMIT fps=%.1f target=%.1f", submitted_count / (now - submit_stats_started), input_pacer.fps)
                submitted_count = 0
                submit_stats_started = now
    finally:
        pose_sender.close()
        try:
            client.close()
        finally:
            active_session_path.unlink(missing_ok=True)
        LOG.info("STATUS DISCONNECTED")
