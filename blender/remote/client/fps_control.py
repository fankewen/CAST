"""Shared input rate configuration and no-catch-up submission pacing."""
import json
import math
import time
from pathlib import Path


def valid_fps(value):
    value = float(value)
    if not math.isfinite(value) or not 1 <= value <= 120:
        raise ValueError("Input FPS must be between 1 and 120")
    return value


class FpsConfigWatcher:
    def __init__(self, path, initial):
        self.path = Path(path) if path is not None else None
        self.fps = valid_fps(initial)
        self.next_poll = 0.0
        self.stamp = self.path.stat().st_mtime_ns if self.path and self.path.exists() else None

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        if self.path is None or now < self.next_poll:
            return self.fps
        self.next_poll = now + 0.5
        try:
            stamp = self.path.stat().st_mtime_ns
            if stamp != self.stamp:
                self.fps = valid_fps(json.loads(self.path.read_text(encoding="utf-8"))["fps"])
                self.stamp = stamp
        except (OSError, ValueError, TypeError, KeyError):
            pass  # Keep the last valid rate during a failed/partial external edit.
        return self.fps


class SubmissionPacer:
    def __init__(self, fps):
        self.fps = valid_fps(fps)
        self.deadline = 0.0

    def update(self, fps):
        fps = valid_fps(fps)
        if fps != self.fps:
            self.fps = fps
            self.deadline = 0.0

    def delay(self, now):
        return max(0.0, self.deadline - now)

    def submitted(self, now):
        # A stall never creates credit for a burst of stale frames.
        self.deadline = now + 1.0 / self.fps
