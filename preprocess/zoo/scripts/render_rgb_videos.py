#!/usr/bin/env python3
"""Run the bundled RGB renderer with portable FFmpeg image-sequence encoding.

The bundled renderer's glob input is unsupported by some Windows FFmpeg
builds. Its Blender rendering and camera settings remain unchanged here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
sys.path.insert(0, str(THIRD_PARTY))
from preprocess import render_bvh_videos_fast as renderer


def encode_numbered_images(image_dir, video, fps=30, crf=18, preset="slow"):
    image_dir = Path(image_dir)
    for extension in ("png", "jpg", "jpeg"):
        frames = sorted(image_dir.glob(f"*.{extension}"))
        if frames:
            break
    if not frames:
        raise FileNotFoundError(f"No rendered RGB frames in {image_dir}")
    pattern = f"%0{len(frames[0].stem)}d.{extension}"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
        "-i", str(image_dir / pattern), "-c:v", "libx264", "-crf", str(crf),
        "-preset", preset, "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(video),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    renderer.convert_images_to_video = encode_numbered_images
    renderer.main()
