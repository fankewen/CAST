#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从一段录像里自动估算镜头畸变，写出 camera_profile.json。

原理(plumb-line)：现实中的直线在画面里被镜头畸变弯成弧线。
把边缘检测出的长链做去畸变，让它们尽量共线，即可反解出畸变系数。
注意：f 与 k1 在像素空间耦合(只有 c=k1/f² 可辨识)，所以 f 由视场角先验给出，
不影响实际矫正效果（f 只决定去畸变后画面的整体缩放，而缩放又会被中心裁切吃掉）。

用法:
    python calibrate_from_video.py recordings\\rec_xxx.mp4 --camera 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import lens
from client_config import default_config_path

SCRIPT_DIR = Path(__file__).resolve().parent


def frames_from_video(path: str, max_frames: int = 12):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"打不开录像: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out, i = [], 0
    step = max(1, total // (max_frames * 4)) if total else 1
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    # 均匀取 max_frames 张
    if len(out) > max_frames:
        idx = np.linspace(0, len(out) - 1, max_frames).astype(int)
        out = [out[k] for k in idx]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="从录像自动估算镜头畸变")
    ap.add_argument("video", help="录像文件路径")
    ap.add_argument("--camera", type=int, default=None, help="该档案对应的摄像头编号")
    ap.add_argument("--fov", type=float, default=lens.DEFAULT_FOV_DEG,
                    help=f"水平视场角先验(度)，默认 {lens.DEFAULT_FOV_DEG}")
    ap.add_argument("--full-width", type=int, default=1280,
                    help="传感器的完整宽度，用来把视场角换算成焦距，默认 1280")
    ap.add_argument("--target-size", type=str, default="1280x720",
                    help="档案对应的采集分辨率，默认 1280x720")
    ap.add_argument("--strength-hint", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=str, default=str(default_config_path().parent / lens.PROFILE_NAME))
    args = ap.parse_args(argv)

    grays = frames_from_video(args.video)
    if not grays:
        raise SystemExit("读不到帧")
    hv, wv = grays[0].shape
    print(f"录像: {args.video}")
    print(f"帧尺寸: {wv}x{hv}   用于标定: {len(grays)} 帧")

    chains = lens.collect_chains(grays)
    print(f"提取长链: {len(chains)} 条")
    if len(chains) < 5:
        print("!! 长链太少，画面里缺少直线结构，估算可能不准")

    f = lens.fov_to_f(args.fov, args.full_width)
    print(f"焦距先验: {f:.1f}px  (水平视场 {args.fov:.0f}°, 按 {args.full_width}px 宽算)")

    s0, _ = lens.straightness_score(chains, lens.camera_matrix_from(f, wv, hv), lens.dist_vector(0.0))
    r = lens.estimate_distortion(chains, (wv, hv), f)
    print(f"\n估算结果: k1={r['k1']:+.4f}  k2={r['k2']:+.4f}  f={r['f']:.1f}")
    print(f"直线度: {s0*1000:.3f}e-3 -> {r['score']*1000:.3f}e-3   改善 {(1-r['score']/s0)*100:.1f}%")
    print(f"像素矫正系数 c = k1/f² = {r['c']:.3e}")

    tw, th = (int(v) for v in args.target_size.lower().split("x"))
    K = lens.camera_matrix_from(r["f"], tw, th)
    extra = {"camera": args.camera, "note": f"由 {Path(args.video).name} 自动估算; f 取 {args.fov:.0f}°视场先验",
             "pixel_correction_c": r["c"], "straightness_gain": round(float(1 - r["score"] / s0), 4)}
    lens.save_profile(args.out, K, r["dist_coeffs"], (tw, th), 720, "auto-plumbline", extra)
    print(f"\n已写出: {args.out}")
    print(f"  image_size={tw}x{th}  camera={args.camera}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())