"""Export the six Mixamo rigs, render their clips, and build the paired cache."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("Big_Vegas", "Brian", "Mousey", "Mutant", "The_Boss", "Ty")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--blender", default=os.environ.get("BLENDER_EXECUTABLE", "blender"))
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    log_dir = (args.log_dir or cache.parent / "logs").expanduser().resolve()
    blender = shutil.which(args.blender) or str(Path(args.blender).expanduser().resolve())
    if not Path(blender).is_file():
        raise FileNotFoundError(f"Blender executable not found: {args.blender}")
    if args.workers < 1 or args.threads < 1:
        raise ValueError("--workers and --threads must be positive")
    for name in NAMES:
        if not any((raw_root / name).glob("*.fbx")):
            raise FileNotFoundError(f"No FBX clips in {raw_root / name}")

    xvfb = []
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        executable = shutil.which("xvfb-run")
        if executable is None:
            raise FileNotFoundError("Headless rendering requires xvfb-run")
        xvfb = [executable, "-a"]
    cache.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(name: str, clip: str | None = None) -> None:
        key = f"{name}__{clip or 'export'}"
        if clip and (cache / name / "rgba" / clip / "DONE").is_file():
            print("EXISTING", key, flush=True)
            return
        command = [
            *xvfb, blender, "-b", "-t", str(args.threads),
            "--python-exit-code", "1", "--python", str(ROOT / "scripts" / "prepare_mixamo.py"),
            "--", "--root", str(raw_root), "--out", str(cache), "--character", name,
        ]
        if clip:
            command += ["--clip", clip]
        log_path = log_dir / f"{key}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Mixamo {key} failed; see {log_path}")
        print("DONE", key, flush=True)

    for name in NAMES:
        character = cache / name
        if not (character / "meta.json").is_file() or not (character / "target_rest.blend").is_file():
            run(name)
    jobs = [
        (name, motion["clip_name"])
        for name in NAMES
        for motion in json.loads((cache / name / "meta.json").read_text(encoding="utf-8"))["motions"]
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, *job) for job in jobs]):
            future.result()

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "finalize_dataset.py"), "--cache", str(cache)],
        cwd=ROOT,
        check=True,
    )
    print("ALL_CACHE_COMPLETE", cache, flush=True)


if __name__ == "__main__":
    main()
