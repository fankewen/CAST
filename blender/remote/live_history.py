"""Bounded, nonblocking inference -> image-history recorder.

The GPU thread only samples/copies selected mask data and enqueues owned CPU
arrays. All image encoding, filesystem work, indexing and retention happen here.
"""
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

LOG = logging.getLogger("cast.history")
KINDS = ("original", "bbox", "sam_mask", "mask_overlay", "crop", "model_input_256")


class LiveHistory:
    def __init__(self, home):
        self.root = Path(home) / "live_history"
        self.root.mkdir(exist_ok=True, mode=0o700)
        self.enabled = os.environ.get("CAST_HISTORY_ENABLED", "0") == "1"
        self.interval = max(0.25, float(os.environ.get("CAST_HISTORY_INTERVAL_SECONDS", "1")))
        self.retention = max(60, int(os.environ.get("CAST_HISTORY_RETENTION_SECONDS", "86400")))
        self.max_bytes = max(10*1024**2, int(os.environ.get("CAST_HISTORY_MAX_BYTES", str(2*1024**3))))
        self.max_samples = max(10, int(os.environ.get("CAST_HISTORY_MAX_SAMPLES", "3600")))
        self.queue = queue.Queue(maxsize=4)
        self.stopping = threading.Event()
        self.last_capture = {}
        self.capture_drops = 0
        self.write_errors = 0
        self.total_saved_run = 0
        self.records = []
        self.bytes = 0
        self.started = time.time()
        self.thread = threading.Thread(target=self.run, daemon=True, name="cast-image-history")
        self.thread.start()

    def new_stream(self):
        # Independent archive segment for each reset, never reuse seq filenames.
        self.last_capture.clear()
        return uuid.uuid4().hex

    def due(self, stream):
        if not self.enabled:
            return False
        now = time.monotonic()
        if now-self.last_capture.get(stream, -1e20) < self.interval:
            return False
        self.last_capture[stream] = now
        if self.queue.full():
            self.capture_drops += 1
            return False
        return True

    def submit(self, record):
        try:
            self.queue.put_nowait(record)
            return True
        except queue.Full:
            self.capture_drops += 1
            return False

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=10)

    def atomic_json(self, path, data):
        temp = path.with_suffix(".tmp")
        with temp.open("w") as f:
            json.dump(data, f, separators=(",", ":"), allow_nan=False)
        os.chmod(temp, 0o600)
        temp.replace(path)

    def restore(self):
        try:
            data = json.loads((self.root / "index.json").read_text())
            self.records = [r for r in data["records"]
                            if re.fullmatch(r"[0-9a-f]{32}", r["id"])
                            and (self.root / r["id"] / "meta.json").is_file()]
        except (OSError, ValueError, KeyError):
            self.records = []
        # Recover completed samples published just before an interrupted index
        # update. Remove only recorder-owned interrupted directories.
        indexed = {r["id"] for r in self.records}
        for folder in self.root.iterdir():
            if folder.is_dir() and re.fullmatch(r"[0-9a-f]{32}", folder.name) and folder.name not in indexed:
                try:
                    r = json.loads((folder / "meta.json").read_text())["summary"]
                    assert r["id"] == folder.name
                    self.records.append(r)
                except (OSError, ValueError, KeyError, AssertionError):
                    shutil.rmtree(folder)
        self.records.sort(key=lambda r: r["ts"])
        self.bytes = sum(r["bytes"] for r in self.records)
        self.prune()
        self.publish()

    def prune(self):
        deadline = time.time()-self.retention
        while self.records and (self.records[0]["ts"] < deadline or self.bytes > self.max_bytes
                                or len(self.records) > self.max_samples):
            oldest = self.records[0]
            shutil.rmtree(self.root / oldest["id"], ignore_errors=False)
            self.records.pop(0)
            self.bytes -= oldest["bytes"]

    def publish(self):
        self.atomic_json(self.root / "index.json", {
            "updated_at": time.time(), "enabled": self.enabled, "started_at": self.started,
            "interval_seconds": self.interval, "retention_seconds": self.retention,
            "max_bytes": self.max_bytes, "max_samples": self.max_samples,
            "bytes": self.bytes, "saved_this_run": self.total_saved_run,
            "queue_depth": self.queue.qsize(), "queue_capacity": 4,
            "dropped_this_run": self.capture_drops, "write_errors_this_run": self.write_errors,
            "records": self.records,
        })

    def write(self, record):
        # Retention is independent of incoming request names/rig IDs.
        self.prune()
        rid = uuid.uuid4().hex
        folder = self.root / rid
        folder.mkdir(mode=0o700)
        try:
            image = Image.fromarray(record.pop("rgb"))
            mask = record.pop("mask", None)
            box = tuple(record["bbox"]) if record["bbox"] is not None else None
            image.save(folder / "original.jpg", quality=90)
            overlay = image.copy()
            if box is not None:
                ImageDraw.Draw(overlay).rectangle(box, outline=(230, 64, 45), width=3)
            else:
                ImageDraw.Draw(overlay).text((10,10),'NO PERSON - MOTION HELD',fill=(230,64,45))
            overlay.save(folder / "bbox.jpg", quality=90)
            available = ["original", "bbox"]
            if box is not None:
                crop = image.crop(box)
                crop.save(folder / "crop.jpg", quality=90)
                crop.resize((256, 256), Image.Resampling.BICUBIC).save(folder / "model_input_256.png")
                available += ["crop", "model_input_256"]
            if mask is not None:
                mask_image = Image.fromarray(mask.astype(np.uint8)*255)
                mask_image.save(folder / "sam_mask.png")
                region = np.array(mask_image.resize(image.size, Image.Resampling.NEAREST)) > 0
                tinted = np.array(image)
                tinted[region] = (tinted[region]*.5+np.array([0, 255, 0])*.5).astype(np.uint8)
                Image.fromarray(tinted).save(folder / "mask_overlay.jpg", quality=90)
                available += ["sam_mask", "mask_overlay"]
            summary = {
                "id": rid, "stream": record["stream"], "session_label": record["session_label"],
                "ts": record["ts"], "seq": record["seq"], "shape": [image.height, image.width],
                "bbox": list(box) if box is not None else None, "rig_id": record["rig_id"], "joints": record["joints"],
                "tracking": record.get('tracking'), "motion_valid": record.get('motion_valid',True),
                "model": record.get('model'),
                "server_ms": record["timing_ms"]["server_total"], "available": available,
                "bytes": 0,
            }
            meta = {"summary": summary, "timestamp_ms": record["timestamp_ms"],
                    "timing_ms": record["timing_ms"],
                    "motion": record["motion"],
                    "source": "online_8080_successful_frame",
                    "original_note": "服务解码后的RGB重新编码为JPEG90，非上传文件原始字节；模型输入为归一化前RGB。",
                    "mask_note": "SAM编码器阈值mask；human模式crop受人体检测框约束或回退，两者不一定同区域；无目标时不存crop和mask。"}
            self.atomic_json(folder / "meta.json", meta)
            summary["bytes"] = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())+128
            self.atomic_json(folder / "meta.json", meta)
            for p in folder.iterdir():
                os.chmod(p, 0o600)
            self.records.append(summary)
            self.bytes += summary["bytes"]
            self.total_saved_run += 1
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def run(self):
        try:
            self.restore()
        except Exception:
            LOG.exception("Image history initialization failed; inference remains available")
            self.write_errors += 1
        last_publish = 0
        while not self.stopping.is_set() or not self.queue.empty():
            item = None
            try:
                item = self.queue.get(timeout=1)
                self.write(item)
            except queue.Empty:
                pass
            except Exception:
                self.write_errors += 1
                LOG.exception("Image history write failed; inference unaffected")
            finally:
                if item is not None:
                    self.queue.task_done()
            if item is not None or time.monotonic()-last_publish >= 5:
                try:
                    self.prune()
                    self.publish()
                except Exception:
                    self.write_errors += 1
                    LOG.exception("Image history index/retention failed")
                last_publish = time.monotonic()
