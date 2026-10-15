#!/usr/bin/env python3
"""Long-lived CAST engine process driven by the Blender panel.

Torch never enters Blender: the panel sends JSON commands over a local socket
and this process owns the checkpoint, the skeleton conditioning, the single
compile, and the 30 FPS feed.

Commands (one JSON object per line, one reply line per command):

``load_weights``  config, checkpoint           -> network + backbone compiled
``load_rig``      static, bind_transforms      -> skeleton state + streaming step compiled + SAM2.1 loaded
``run_video``     video, fps, after_end         -> starts the uniform feed (replies at once); also swaps
                                                  the clip while one is running or paused
``pause_feed``                                 -> holds the feed at the current frame (no poses sent)
``resume_feed``                                -> carries on from the frame it held at
``stop_feed``                                  -> stops the feed, keeps the model
``status``                                     -> current phase
``shutdown``                                   -> exit

The two load steps are separate on purpose: the panel has one button for
"load weights" and another for "load model", and the second one is where the
skeleton-only state, the streaming graph *and the segmentation model* are
built.  Segmenting is part of every frame, so its model is loaded up front
together with the mocap model rather than lazily on the first frame.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import struct
import sys
import threading
import time
from pathlib import Path

if not os.environ.get("CUDA_HOME"):
    for _cuda_root in (
        os.environ.get("CAST_CUDA_HOME"),
        "/usr/local/cuda",
    ):
        if _cuda_root and (Path(_cuda_root) / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = _cuda_root
            break

import numpy as np
import torch

HEADER = struct.Struct("<II")
POSE_PORT_DEFAULT = 5599


class StreamEngine:
    """CAST streaming engine with an explicit weights step and rig step."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.amp_dtype = args.amp_dtype
        self.repo = args.repo
        self.runtime = Path(args.runtime_dir)
        self.runtime.mkdir(parents=True, exist_ok=True)
        # The deployment object owns the network, the skeleton state and the
        # compiled step; the engine only drives it.
        self.stream = None
        self.cfg = None
        self.joint_count = None
        self.image_size = None
        self.cache_length = int(args.cache_length)
        self.feed_thread = None
        self.feed_stop = threading.Event()
        self.feed_pause = threading.Event()
        self.feed_video = None
        self.feed_frames = 0
        self.phase = "idle"
        self.log_path = self.runtime / "engine.log"

    # ---------------------------------------------------------------- logging
    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def status(self, phase: str) -> None:
        self.phase = phase
        (self.runtime / "status.json").write_text(
            json.dumps({"phase": phase, "time": time.time()}, indent=2), encoding="utf-8"
        )
        self.log(f"PHASE {phase}")

    # ------------------------------------------------------------ phase 1
    def load_weights(
        self,
        config: str,
        checkpoint: str,
        sam_cfg: str | None = None,
        sam_ckpt: str | None = None,
    ) -> dict:
        from inference.deploy_realtime_video2motion import RealtimeVideo2Motion

        self.status("loading_weights")
        # The RGB backbone comes from the config itself (target + repo + weights),
        # so it is not a separate input here.
        self.stream = RealtimeVideo2Motion(
            config=config,
            checkpoint=checkpoint,
            device=str(self.device),
            cache_length=self.cache_length,
            amp_dtype=self.amp_dtype,
            compile_cache_dir=self.args.compile_cache_dir,
        )
        self.cfg = self.stream.cfg
        self.image_size = self.stream.image_size
        self.spec = {"config": config, "checkpoint": checkpoint}
        self.sam_cfg = sam_cfg
        self.sam_ckpt = sam_ckpt
        # Every model the feed needs is loaded here: segmentation does not depend
        # on the skeleton, so it no longer waits for "Load Model".
        segmenter = self.load_segmenter()
        self.status("weights_ready")
        return {
            "ok": True,
            "phase": "weights_ready",
            "image_size": self.image_size,
            "segmenter": segmenter,
        }

    # ------------------------------------------------------------ phase 2
    def load_rig(self, static: str, bind_transforms: str | None = None) -> dict:
        if self.stream is None:
            raise RuntimeError("load_weights first")
        self.status("loading_rig")
        self.stream.prepare(static)
        self.joint_count = self.stream.joint_count
        # Compile against a placeholder reference: the real first frame of a clip
        # is copied into the graph later by ``bind_reference``.
        self.stream.compile_streaming()
        self.status("model_ready")
        return {
            "ok": True,
            "phase": "model_ready",
            "joints": int(self.joint_count),
            "cache_length": self.cache_length,
            "static": static,
            "bind_transforms": bind_transforms,
        }

    # ------------------------------------------------------------ phase 3
    def run_video(self, video: str, fps: float, after_end: str, pose_port: int) -> dict:
        if self.stream is None:
            raise RuntimeError("load_rig first")
        if self.feed_thread is not None and self.feed_thread.is_alive():
            # A feed is already running: hand over to the new clip. This is also
            # the path taken when a new video is picked while the feed is paused.
            self.feed_stop.set()
            self.feed_thread.join(timeout=30)
        # A new clip starts its own temporal window instead of inheriting the
        # previous video's streaming state.
        self.stream.reset_streaming()
        self.feed_stop = threading.Event()
        paused = self.feed_pause.is_set()
        self.feed_video = video
        self.feed_frames = 0
        self.feed_thread = threading.Thread(
            target=self._feed,
            args=(video, float(fps), after_end, int(pose_port)),
            daemon=True,
        )
        self.feed_thread.start()
        self.log(f"VIDEO_LOADED {video} paused={paused}")
        return {
            "ok": True,
            "phase": "feed_paused" if paused else "feeding",
            "video": video,
            "paused": paused,
        }

    def pause_feed(self) -> dict:
        """Hold the feed where it is: no new frames, no poses, the mesh stays put."""
        self.feed_pause.set()
        self.log("PAUSE_REQUESTED")
        return {
            "ok": True,
            "phase": "feed_paused",
            "video": self.feed_video,
            "frames": self.feed_frames,
        }

    def resume_feed(self) -> dict:
        self.feed_pause.clear()
        self.log("RESUME_REQUESTED")
        return {
            "ok": True,
            "phase": "feeding",
            "video": self.feed_video,
            "frames": self.feed_frames,
        }

    def stop_feed(self) -> dict:
        self.feed_stop.set()
        self.feed_pause.clear()
        if self.feed_thread is not None:
            self.feed_thread.join(timeout=30)
        self.feed_video = None
        self.status("model_ready")
        return {"ok": True, "phase": "model_ready"}

    def _feed(self, video: str, fps: float, after_end: str, pose_port: int) -> None:
        from inference.video2motion_unlabeled import iter_video_rgb, video_metadata

        sender = PoseSocket(pose_port, self.log)
        try:
            metadata = video_metadata(video)
            frame_count = int(metadata["source_frame_count"])
            self.log(
                f"FEED_START frames={frame_count} fps={fps} after_end={after_end} "
                f"pose_port={pose_port}"
            )
            self.status("feeding")

            def make_source():
                # One SAM2.1 forward per frame, in order, on this thread.
                from inference.realtime_mask import CausalCrops

                return CausalCrops(
                    iter_video_rgb(video, frame_count), self.segmenter(), size=self.image_size
                )

            with torch.inference_mode():
                source = make_source()
                # The first frame doubles as the streaming reference and as the
                # input of slot 0, exactly like every later frame does.
                frame_now = next(source)
                self.stream.bind_reference(frame_now)
                period = 1.0 / fps
                clock = time.perf_counter()
                pushed = 0
                steps = []
                announced = False
                held = False
                while not self.feed_stop.is_set():
                    if self.feed_pause.is_set():
                        # Paused: no stepping, no poses, nothing pulled from the
                        # video. The pacing clock is rebased every tick so that
                        # resuming continues from here instead of bursting to
                        # catch up on the time spent paused.
                        if not held:
                            held = True
                            self.log(f"FEED_HELD at frame {pushed}")
                        time.sleep(0.05)
                        clock = time.perf_counter() - pushed * period
                        continue
                    if held:
                        held = False
                        self.log(f"FEED_RESUMED from frame {pushed}")
                    target = clock + pushed * period
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    # Strictly serial, one frame at a time:
                    #   crop of frame k exists -> encode it -> step the model
                    #   -> send the pose -> only then ask for frame k + 1.
                    stamp = time.perf_counter()
                    pose = self.stream.step(frame_now)
                    steps.append(time.perf_counter() - stamp)
                    sender.send(pose)
                    pushed += 1
                    self.feed_frames = pushed
                    if pushed % max(1, int(fps)) == 0:
                        recent = np.asarray(steps[-int(fps):]) * 1000
                        extra = ""
                        stats = getattr(source, "stats", None)
                        if callable(stats):
                            values = stats()
                            if values:
                                extra = (
                                    f" sam_mean={values['sam_ms_mean']:.1f}ms"
                                    f" crop_mean={values['crop_ms_mean']:.1f}ms"
                                    f" seg_mean={values['seg_ms_mean']:.1f}ms"
                                )
                        self.log(
                            f"FEED frame={pushed} step_mean={recent.mean():.1f}ms "
                            f"step_max={recent.max():.1f}ms{extra}"
                        )
                    try:
                        frame_now = next(source)
                    except StopIteration:
                        if after_end == "loop":
                            if not announced:
                                announced = True
                                self.log("VIDEO_ENDED, restarting the video")
                            source = make_source()
                            frame_now = next(source)
                        else:
                            if not announced:
                                announced = True
                                self.log("VIDEO_ENDED, repeating the last frame")
                self.log(f"FEED_STOPPED after {pushed} frames")
        except Exception as exc:  # noqa: BLE001
            self.log(f"FEED_ERROR {type(exc).__name__}: {exc}")
            self.status("feed_error")
        finally:
            sender.close()

    def load_segmenter(self) -> dict:
        """Build SAM 2.1 from the cfg/ckpt the panel sent with ``load_weights``.

        It runs in the weights step rather than in "Load Model" because
        segmentation does not depend on the skeleton, and loading it here instead
        of on the first feed frame keeps the start of the feed clean: the weights
        are resident and the warm-up forward below materialises every kernel, so
        the first real frame costs the same as the hundredth one.
        """
        if getattr(self, "_segmenter", None) is not None:
            return {
                "loaded": True,
                "cached": True,
                "model": getattr(self, "segmenter_name", "?"),
                "image_size": int(self.args.seg_image_size),
            }

        from inference.realtime_mask import CausalBoxSegmenter

        config = getattr(self, "sam_cfg", None) or self.args.sam2_config
        checkpoint = getattr(self, "sam_ckpt", None) or self.args.sam2_checkpoint
        if not Path(checkpoint).is_file():
            raise RuntimeError(f"SAM checkpoint not found: {checkpoint}")
        self.status("loading_segmenter")
        self.log(
            f"SEGMENTER sam2.1 cfg={config} image_size={self.args.seg_image_size} "
            f"checkpoint={checkpoint}"
        )
        begin = time.perf_counter()
        segmenter = CausalBoxSegmenter(
            config,
            checkpoint,
            device=str(self.device),
            amp_dtype=self.amp_dtype,
            image_size=self.args.seg_image_size,
        )
        # The constructor already captured the CUDA graph; one real frame on top
        # of that walks the numpy/cv2 part of the path too, so the first frame of
        # the feed costs the same as the hundredth.
        width, height = (int(value) for value in str(self.args.warmup_shape).split("x"))
        segmenter.box(np.zeros((height, width, 3), dtype=np.uint8))
        segmenter.reset()
        self._segmenter = segmenter
        self.segmenter_name = Path(checkpoint).name
        seconds = time.perf_counter() - begin
        self.log(f"SEGMENTER ready in {seconds:.1f}s (CUDA graph captured, {width}x{height})")
        return {
            "loaded": True,
            "model": self.segmenter_name,
            "image_size": int(self.args.seg_image_size),
            "config": config,
            "checkpoint": checkpoint,
            "seconds": round(seconds, 1),
        }

    def segmenter(self):
        """The segmenter built by ``load_weights``; builds it if that was skipped."""
        if getattr(self, "_segmenter", None) is None:
            self.log("SEGMENTER missing, building it now (the Weights box is used)")
            self.load_segmenter()
        return self._segmenter


class PoseSocket:
    """Sends length-prefixed rot6d frames to the Blender panel (reconnects)."""

    def __init__(self, port: int, log):
        self.port = port
        self.log = log
        self.socket = None
        self.attempts = 0

    def _connect(self) -> bool:
        try:
            self.socket = socket.create_connection(("127.0.0.1", self.port), timeout=5.0)
            self.log(f"POSE_CONNECTED port={self.port}")
            return True
        except OSError:
            self.socket = None
            self.attempts += 1
            if self.attempts % 20 == 1:
                self.log(f"POSE_WAITING port={self.port} attempts={self.attempts}")
            return False

    def send(self, rot6d: torch.Tensor) -> None:
        values = rot6d.detach().float().cpu().numpy().reshape(-1)
        payload = HEADER.pack(0, values.size // 6) + struct.pack(f"<{values.size}f", *values.tolist())
        for _ in range(2):
            if self.socket is None and not self._connect():
                time.sleep(0.2)
                continue
            try:
                self.socket.sendall(payload)
                return
            except OSError:
                self.socket = None
        time.sleep(0.05)

    def close(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass
            self.socket = None


class ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        engine: StreamEngine = self.server.engine
        for raw in self.rfile:
            raw = raw.strip()
            if not raw:
                continue
            try:
                request = json.loads(raw.decode("utf-8"))
                reply = dispatch(engine, request)
            except Exception as exc:  # noqa: BLE001
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                engine.log(f"COMMAND_ERROR {reply['error']}")
            self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))
            self.wfile.flush()
            if request.get("cmd") == "shutdown":
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                break


def dispatch(engine: StreamEngine, request: dict) -> dict:
    cmd = request.get("cmd")
    engine.log(f"COMMAND {cmd}")
    if cmd == "load_weights":
        return engine.load_weights(request["config"], request["checkpoint"])
    if cmd == "load_rig":
        return engine.load_rig(request["static"], request.get("bind_transforms"))
    if cmd == "run_video":
        return engine.run_video(
            request["video"],
            request.get("fps", 30.0),
            request.get("after_end", "repeat-last"),
            request.get("pose_port", POSE_PORT_DEFAULT),
        )
    if cmd == "pause_feed":
        return engine.pause_feed()
    if cmd == "resume_feed":
        return engine.resume_feed()
    if cmd == "stop_feed":
        return engine.stop_feed()
    if cmd == "status":
        return {"ok": True, "phase": engine.phase}
    if cmd == "shutdown":
        return {"ok": True, "phase": "shutdown"}
    return {"ok": False, "error": f"unknown cmd {cmd!r}"}


class ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser(description="CAST engine control server")
    _repo_root = str(Path(__file__).resolve().parents[1])
    parser.add_argument("--repo", default=_repo_root)
    parser.add_argument("--runtime-dir", default=str(Path(_repo_root) / "runtime"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=5598)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--cache-length", type=int, default=81)
    parser.add_argument("--compile-cache-dir", default="/path/to/cache/cast_realtime_compile_live")
    parser.add_argument(
        "--seg-image-size",
        type=int,
        default=512,
        help="encoder resolution (1024 = official). 512 is the measured trade-off",
    )
    parser.add_argument(
        "--warmup-shape",
        default="512x512",
        help="WxH of the synthetic frame used to warm the segmenter while loading",
    )
    # SAM defaults; the panel's Weights box normally sends both with load_weights.
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM 2.1 config name, resolved inside the SAM2 install",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default="/path/to/sam2.1_hiera_tiny.pt",
        help="SAM 2.1 weights",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    engine = StreamEngine(args)
    engine.status("idle")
    server = ControlServer((args.host, args.control_port), ControlHandler)
    server.engine = engine
    print(f"ENGINE_LISTENING {args.host}:{args.control_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        engine.stop_feed()
        server.server_close()
        print("ENGINE_EXIT", flush=True)


if __name__ == "__main__":
    main()
