"""Recorder unit tests in temporary directories, no live images or GPU needed."""
import json
import os
os.environ["CAST_HISTORY_ENABLED"]="1"
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch
import numpy as np
from PIL import Image
from live_history import LiveHistory


def sample(stream, seq=0):
    return {"stream": stream, "session_label": "testonly", "ts": time.time(),
            "seq": seq, "timestamp_ms": 123, "rgb": np.full((64, 96, 3), 123, dtype=np.uint8),
            "mask": np.ones((32, 32), dtype=bool), "bbox": [10, 8, 70, 60],
            "rig_id": "../../not-a-path", "joints": 2,
            "timing_ms": {"decode": 1, "sam_crop": 2, "model_and_d2h": 3, "server_total": 6},
            "motion": {"rot6d": [[1, 0, 0, 0, 1, 0]]*2}}


class RecorderTests(unittest.TestCase):
    def test_write_restore_and_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = LiveHistory(temp)
            stream = recorder.new_stream()
            self.assertTrue(recorder.due(stream))
            self.assertFalse(recorder.due(stream))
            self.assertTrue(recorder.submit(sample(stream)))
            recorder.queue.join()
            recorder.close()
            index = json.loads((Path(temp)/"live_history/index.json").read_text())
            self.assertEqual(len(index["records"]), 1)
            r = index["records"][0]
            folder = Path(temp)/"live_history"/r["id"]
            with Image.open(folder/"model_input_256.png") as image:
                self.assertEqual(image.size, (256, 256))
            self.assertEqual(len(r["available"]), 6)
            self.assertEqual(json.loads((folder/"meta.json").read_text())["source"],
                             "online_8080_successful_frame")
            restored = LiveHistory(temp)
            restored.close()
            self.assertEqual(len(restored.records), 1)
            restored.max_samples = 1
            restored.write(sample(uuid.uuid4().hex, 1))
            restored.prune()
            self.assertEqual(len(restored.records), 1)
            self.assertFalse(folder.exists())
            restored.max_bytes = 1
            restored.prune()
            self.assertEqual(restored.records, [])
            self.assertEqual(restored.bytes, 0)

    def test_queue_full_nonblocking(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = LiveHistory(temp)
            recorder.close()
            for i in range(4):
                self.assertTrue(recorder.submit(sample("a"*32, i)))
            before = time.monotonic()
            self.assertFalse(recorder.submit(sample("a"*32)))
            self.assertLess(time.monotonic()-before, .1)
            self.assertEqual(recorder.capture_drops, 1)

    def test_failure_isolated_and_disabled(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"CAST_HISTORY_ENABLED": "0"}):
            recorder = LiveHistory(temp)
            self.assertFalse(recorder.due(recorder.new_stream()))
            recorder.close()
            bad = sample("a"*32)
            bad["mask"] = np.zeros((3, 3, 3, 3))
            with self.assertRaises(Exception):
                recorder.write(bad)
            self.assertFalse(any(p.is_dir() for p in recorder.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
