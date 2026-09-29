#!/usr/bin/env python3
"""Export Zoo FBXs as cache source assets.

Original BVH files are not accepted as input. Required BVHs are generated from
the character and animation FBXs in the supplied Zoo directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


VIEWS = (0, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270, 315)


def character_base(species: str, fbxs: list[Path]) -> Path:
    for fbx in fbxs:
        if fbx.stem in (species, species + "ALL"):
            return fbx
    for fbx in fbxs:
        if "-" not in fbx.stem and "TPOSE" not in fbx.stem.upper():
            return fbx
    return min(fbxs, key=lambda path: len(path.name))


def character_candidates(species: str, fbxs: list[Path], animation: Path) -> list[Path]:
    base = character_base(species, fbxs)
    key = re.sub(r"[^a-z0-9]", "", species.casefold())
    preferred = [path for path in fbxs if path != animation and path != base
                 and re.sub(r"[^a-z0-9]", "", path.stem.casefold()) in (key, key + "all")]
    others = [path for path in fbxs if path != animation and path != base
              and path not in preferred and "-" not in path.stem
              and "TPOSE" not in path.stem.upper()]
    return [base] + preferred + others + [animation]


def canonicalize_blender_bvh(path: Path) -> None:
    """Remove Blender 5.x's duplicate child translations without changing pose.

    Some FBXs import with several top-level bones. Blender then emits a
    synthetic 0-channel root and 6-channel children. In the Truebone files the
    child position channels duplicate their fixed OFFSETs; the bundled BVH
    reader expects a 6-channel root and 3-channel children.
    """
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    motion_line = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
    channels = []
    for i, line in enumerate(lines[:motion_line]):
        parts = line.strip().split()
        if parts[:1] == ["CHANNELS"]:
            channels.append((i, int(parts[1])))
    if not channels:
        raise ValueError(f"No BVH channels: {path}")
    if channels[0][1] == 6 and all(row[1] == 3 for row in channels[1:]):
        return
    if channels[0][1] not in (0, 6) or any(row[1] not in (3, 6) for row in channels[1:]):
        raise ValueError(f"Unsupported FBX BVH channel layout: {path}")
    data_start = motion_line + 3
    frame_rows = [i for i in range(data_start, len(lines)) if lines[i].strip()]
    for row in frame_rows:
        values = np.fromstring(lines[row], sep=" ")
        result = []
        cursor = 0
        for joint, (_, count) in enumerate(channels):
            block = values[cursor:cursor + count]
            cursor += count
            if joint == 0 and count == 0:
                result.extend([0.0] * 6)
            elif joint and count == 6:
                result.extend(block[3:])
            else:
                result.extend(block)
        lines[row] = " ".join(f"{x:.6f}" for x in result) + "\n"
    for joint, (i, count) in enumerate(channels):
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        if joint == 0 and count == 0:
            lines[i] = indent + "CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
        elif joint and count == 6:
            lines[i] = indent + "CHANNELS 3 Xrotation Yrotation Zrotation\n"
    path.write_text("".join(lines), encoding="utf-8")


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prepare_one(animation: Path, candidates: list[Path], output: Path,
                blender: Path, third_party: Path) -> dict:
    species = animation.parent.name
    motion = f"{species}#{animation.stem.replace(' ', '_')}"
    mapping = {
        "animation_fbx": str(animation), "motion": motion,
        "skeleton_source": "animation FBX; character mesh and weights from base FBX",
    }

    driver = Path(__file__).with_name("extract_fbx_assets_blender.py")
    run([
        str(blender), "-b", "--python-exit-code", "1", "--python", str(driver), "--",
        "--animation-fbx", str(animation), "--base-candidates", json.dumps([str(path) for path in candidates]),
        "--motion", motion, "--species", species, "--output-root", str(output),
    ])
    mapping["base_fbx"] = json.loads((output / "fbx_selection" / f"{motion}.json").read_text(encoding="utf-8"))["base_fbx"]

    from utils import bvh as BVH
    from utils.bvh_tools import face_forward_bvh_with_scale

    exported = output / "motions" / (motion + ".bvh")
    canonicalize_blender_bvh(exported)
    canonicalize_blender_bvh(output / "characters_by_motion" / motion / "rest.bvh")
    animation_data, names, frametime = BVH.load(str(exported))
    source_scale = 1.0 if motion == "Monkey#MonkeyAll" else 0.01
    face_forward_bvh_with_scale(str(exported), animation_data, names, frametime, scale=source_scale)
    _, prepared_names, _ = BVH.load(str(exported))
    mapping["fbx_exported_joints"] = len(prepared_names)
    mapping["fbx_joint_names"] = prepared_names
    from scipy.spatial.transform import Rotation
    bvh_dir = output / "bvh" / motion
    bvh_dir.mkdir(parents=True)
    shutil.copyfile(exported, bvh_dir / "y0.bvh")
    for view in VIEWS[1:]:
        anim, joint_names, ft = BVH.load(str(exported))
        root = Rotation.from_quat(np.asarray(anim.rotations[:, 0])[:, [1, 2, 3, 0]])
        turn = Rotation.from_euler("y", view, degrees=True)
        root_new = turn * root
        anim.rotations.qs[:, 0] = np.asarray(root_new.as_quat())[:, [3, 0, 1, 2]]
        anim.positions[:, 0] = turn.apply(anim.positions[:, 0])
        BVH.save(str(bvh_dir / f"y{view}.bvh"), anim, joint_names, ft)
    print(f"Prepared {motion}: {len(prepared_names)} joints")
    return mapping


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zoo-root", type=Path, required=True, help="Truebone Zoo directory containing character FBXs")
    p.add_argument("--output-root", type=Path, required=True, help="Fresh prepared Zoo source directory")
    p.add_argument("--blender", type=Path, required=True)
    p.add_argument("--third-party-root", type=Path, default=Path(__file__).resolve().parents[1] / "third_party")
    args = p.parse_args()

    root = args.zoo_root.resolve()
    if not root.is_dir():
        p.error(f"--zoo-root must be a Truebone Zoo directory of FBXs; original BVH files are not accepted: {root}")
    jobs = []
    for species_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        fbxs = sorted(path for path in species_dir.iterdir() if path.suffix.lower() == ".fbx")
        if not fbxs:
            continue
        base = character_base(species_dir.name, fbxs)
        jobs.extend((animation, character_candidates(species_dir.name, fbxs, animation)) for animation in fbxs
                    if animation != base and "TPOSE" not in animation.stem.upper())
    if not jobs:
        p.error("No animation FBXs found")
    if not args.blender.is_file():
        p.error(f"Blender does not exist: {args.blender}")

    third_party = args.third_party_root.resolve()
    if not (third_party / "utils" / "bvh_tools.py").is_file():
        p.error(f"Missing third_party/utils/bvh_tools.py: {third_party}")
    sys.path.insert(0, str(third_party))

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    mappings = []
    for index, (animation, candidates) in enumerate(jobs, 1):
        try:
            mapping = prepare_one(animation, candidates, output, args.blender.resolve(), third_party)
        except subprocess.CalledProcessError:
            print(f"Skipped FBX that Blender could not import: {animation}", file=sys.stderr, flush=True)
            continue
        mappings.append(mapping)
        print(f"[{index}/{len(jobs)}] {mapping['motion']}", flush=True)
        (output / "fbx_mapping.json").write_text(json.dumps(mappings, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
