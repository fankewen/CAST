#!/usr/bin/env python3
"""Launch Blender workers to render Zoo masks matching the supplied RGB views."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", required=True, help="Blender executable")
    parser.add_argument("--scene", type=Path, required=True, help="Bundled preprocess/blank.blend")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--displays", nargs="+", help="Optional X displays, assigned round-robin to workers")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be positive")
    manifest = json.loads((args.vertices_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("jobs"):
        parser.error("Vertex extraction must complete successfully first")
    if not args.scene.is_file():
        parser.error("Scene file does not exist")
    groups = defaultdict(list)
    for row in manifest["jobs"]:
        job = {k: row[k] for k in ("motion", "path", "frames", "vertices", "character_folder")}
        species = row["motion"].split("#", 1)[0]
        scale, position = (0.25, 1.5) if species == "Crow" else ((0.5, 1.0) if species == "Jaws" else (1.3, 1.0))
        job.update(view_scale=scale, object_position=position)
        groups[row["character_folder"]].append(job)
    bins = [{"frames": 0, "jobs": []} for _ in range(min(args.workers, len(groups)))]
    for jobs in sorted(groups.values(), key=lambda rows: sum(r["frames"] for r in rows), reverse=True):
        target = min(bins, key=lambda b: b["frames"])
        target["jobs"].extend(jobs)
        target["frames"] += sum(r["frames"] for r in jobs)
    output = args.output_root.resolve()
    # Use a fresh mask tree so changed vertices cannot silently reuse old masks.
    output.mkdir(parents=True, exist_ok=False)
    run = output / "logs"
    run.mkdir()
    processes = []
    results = []
    started = time.time()
    try:
        for index, batch in enumerate(bins):
            jobs_path = run / f"worker_{index:02d}.json"
            jobs_path.write_text(json.dumps(batch["jobs"], indent=2), encoding="utf-8")
            log = (run / f"worker_{index:02d}.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            if args.displays:
                env["DISPLAY"] = args.displays[index % len(args.displays)]
            command = [
                args.blender, "-b", "--threads", str(args.threads), "--python-exit-code", "1",
                "--python", str(Path(__file__).with_name("render_masks_blender.py")), "--",
                "--jobs-json", str(jobs_path), "--scene", str(args.scene.resolve()),
                "--mask-root", str(output), "--worker-id", str(index),
            ]
            try:
                process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
            except BaseException:
                log.close()
                raise
            processes.append((process, log, index))
        for process, log, index in processes:
            results.append({"worker": index, "returncode": process.wait()})
            log.close()
    finally:
        for process, log, _ in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    failed = any(r["returncode"] for r in results)
    report = {"status": "failed" if failed else "complete", "results": results, "seconds": time.time() - started}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
