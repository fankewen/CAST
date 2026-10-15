"""Enable the CAST Live panel for this Blender session.

Deliberately does nothing else: it enables the addon and gives the timeline room
to keep counting (the feed advances one frame per pose).
"""

import os
import sys

import bpy

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
if ADDON_DIR not in sys.path:
    sys.path.insert(0, ADDON_DIR)

try:
    bpy.ops.preferences.addon_enable(module="cast_live")
    print("ADDON_ENABLED", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"ADDON_ENABLE_FAILED {type(exc).__name__}: {exc}", flush=True)

scene = bpy.context.scene
scene.frame_start = 1
# One pose per frame keeps the timeline moving, so it must not run out.
scene.frame_end = 10_000_000
scene.render.fps = 30

print("PANEL_READY", flush=True)
