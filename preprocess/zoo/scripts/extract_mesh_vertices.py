#!/usr/bin/env python3
"""Extract normalized animated vertices using the bundled Zoo skinning code."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def extract_one(task):
    motion, source, output, third_party = task
    try:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        sys.path.insert(0, third_party)
        from utils.mesh import BVH, extract_mesh_from_bvh
        from utils.common import get_diameter

        source, output = Path(source), Path(output)
        species = motion.split("#", 1)[0]
        specific = source / "characters_by_motion" / motion
        character = specific if (specific / "base_mesh.obj").is_file() else source / "characters_fix_facezplus" / species
        weight = character / "skinning_weights.npy"
        if not weight.is_file():
            weight = character / "skin_weights.npy"
        bvh = source / "bvh" / motion / "y0.bvh"
        animation, _, _ = BVH.load(str(bvh))
        diameter, _ = get_diameter(animation.parents, np.linalg.norm(animation.offsets, axis=1))
        vertices = extract_mesh_from_bvh(
            str(bvh), str(character / "base_mesh.obj"), str(output / ".unused"),
            str(weight), scale=1 / diameter, return_arrays=True,
        )[0].astype(np.float32, copy=False)
        destination = output / species / (motion + ".npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, vertices)
        os.replace(temporary, destination)
        return {
            "motion": motion, "ok": True, "frames": len(vertices),
            "vertices": vertices.shape[1], "path": str(destination),
            "character_folder": str(character),
        }
    except Exception as exc:
        return {"motion": motion, "ok": False, "error": repr(exc), "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--third-party-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--motion", action="append", help="Process only these motion directory names")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    source, output, third_party = (p.resolve() for p in (args.source_root, args.output_root, args.third_party_root))
    if not (third_party / "utils" / "mesh.py").is_file():
        parser.error("--third-party-root must contain utils/mesh.py; use the bundled third_party directory")
    available = {p.name for p in (source / "bvh").iterdir() if (p / "y0.bvh").is_file()}
    motions = sorted(set(args.motion) if args.motion else available)
    if not motions or set(motions) - available:
        parser.error("No matching motions, or a requested motion is missing y0.bvh")
    # Fresh output prevents stale manifests from being reused after a failed run.
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    tasks = [(m, str(source), str(output), str(third_party)) for m in motions]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(extract_one, tasks))
    failures = [r for r in results if not r["ok"]]
    manifest = {
        "status": "failed" if failures else "complete",
        "motions": len(motions), "passed": len(results) - len(failures), "failed": len(failures),
        "source_root": str(source), "jobs": results, "failures": failures,
        "seconds": time.time() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("jobs", "failures")}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
