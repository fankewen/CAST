#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""镜头畸变相关的公共代码：档案读写 / 去畸变映射 / 直线(plumb-line)标定打分。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROFILE_NAME = "camera_profile.json"
DEFAULT_FOV_DEG = 70.0          # 通用 USB 摄像头常见水平视场角


# --------------------------------------------------------------------------- #
# 参数换算
# --------------------------------------------------------------------------- #
def fov_to_f(fov_deg: float, width: int) -> float:
    return float((width / 2.0) / np.tan(np.radians(fov_deg) / 2.0))


def f_to_fov(f: float, width: int) -> float:
    return float(np.degrees(2.0 * np.arctan((width / 2.0) / max(f, 1e-6))))


def camera_matrix_from(f: float, width: int, height: int) -> np.ndarray:
    return np.array([[f, 0.0, (width - 1) / 2.0],
                     [0.0, f, (height - 1) / 2.0],
                     [0.0, 0.0, 1.0]], np.float64)


def dist_vector(k1: float, k2: float = 0.0, k3: float = 0.0,
                p1: float = 0.0, p2: float = 0.0) -> np.ndarray:
    return np.array([k1, k2, p1, p2, k3], np.float64).reshape(-1, 1)


# --------------------------------------------------------------------------- #
# 档案读写
# --------------------------------------------------------------------------- #
def load_profile(path) -> dict | None:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    try:
        K = np.array(raw["camera_matrix"], np.float64).reshape(3, 3)
        d = np.array(raw["dist_coeffs"], np.float64).reshape(-1, 1)
        size = (int(raw["image_size"][0]), int(raw["image_size"][1]))
    except Exception:
        return None
    return {"camera_matrix": K, "dist_coeffs": d, "image_size": size,
            "crop_size": int(raw.get("crop_size", 720)),
            "source": raw.get("source", "unknown"),
            "created": raw.get("created", ""),
            "raw": raw}


def save_profile(path, camera_matrix, dist_coeffs, image_size, crop_size,
                 source: str, extra: dict | None = None) -> dict:
    data = {
        "camera_matrix": np.asarray(camera_matrix, np.float64).reshape(3, 3).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, np.float64).reshape(-1).tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "crop_size": int(crop_size),
        "source": source,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        data.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# --------------------------------------------------------------------------- #
# 去畸变映射（只算需要的那块正方形，省算力）
# --------------------------------------------------------------------------- #
def build_crop_maps(camera_matrix, dist_coeffs, image_size, crop_size, zoom: float = 1.0):
    """直接生成作用在"中心正方形裁切区"上的 remap 映射表。

    做法：把目标相机矩阵的主点平移，让输出图像正好等于去畸变后整幅画面的中心裁切区。
    """
    w, h = int(image_size[0]), int(image_size[1])
    crop = int(crop_size)
    K = np.asarray(camera_matrix, np.float64).reshape(3, 3).copy()
    if zoom != 1.0:
        K[0, 0] *= zoom
        K[1, 1] *= zoom
    left = (w - crop) // 2
    top = (h - crop) // 2
    Kc = K.copy()
    Kc[0, 2] -= left
    Kc[1, 2] -= top
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, np.asarray(dist_coeffs, np.float64).reshape(-1, 1), None, Kc,
        (crop, crop), cv2.CV_32FC1)
    # 允许半个像素的浮点误差: 略微越界由 BORDER_REPLICATE 兜住, 肉眼不可见
    bad = float(((mapx < -0.5) | (mapx > w - 0.5) | (mapy < -0.5) | (mapy > h - 0.5)).mean())
    return mapx, mapy, bad


def build_crop_maps_autozoom(camera_matrix, dist_coeffs, image_size, crop_size,
                             max_zoom: float = 1.8, tol: float = 5e-4):
    """如果去畸变后裁剪区会露黑边，就自动轻微放大直到没有黑边。"""
    zoom, mx, my, bad = 1.0, None, None, 1.0
    while zoom <= max_zoom:
        mx, my, bad = build_crop_maps(camera_matrix, dist_coeffs, image_size, crop_size, zoom)
        if bad <= tol:
            return mx, my, zoom, bad
        zoom *= 1.03
    return mx, my, zoom, bad


def build_preview_maps(camera_matrix, dist_coeffs, image_size, out_width: int):
    """给标定界面用：整幅画面的去畸变映射，输出缩放到 out_width 宽。
    源图是全分辨率，输出是缩放后的尺寸，映射表坐标仍在源图坐标系。"""
    w, h = int(image_size[0]), int(image_size[1])
    scale = out_width / float(w)
    out_size = (int(round(w * scale)), int(round(h * scale)))
    K = np.asarray(camera_matrix, np.float64).reshape(3, 3)
    Kp = K.copy()
    Kp[0, :] *= scale
    Kp[1, :] *= scale
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, np.asarray(dist_coeffs, np.float64).reshape(-1, 1), None, Kp,
        out_size, cv2.CV_32FC1)
    return mapx, mapy, out_size


# --------------------------------------------------------------------------- #
# 直线(plumb-line)标定
# --------------------------------------------------------------------------- #
def _link_segments(segs, dist_th=25.0, ang_th=25.0):
    """把 HoughLinesP 出来的短线段按端点邻近 + 方向一致串成长折线。"""
    def ang(s):
        return np.arctan2(s[3] - s[1], s[2] - s[0]) % np.pi

    n = len(segs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        ai = ang(segs[i])
        ei = np.array([[segs[i][0], segs[i][1]], [segs[i][2], segs[i][3]]], float)
        for j in range(i + 1, n):
            da = abs(ai - ang(segs[j]))
            if np.degrees(min(da, np.pi - da)) > ang_th:
                continue
            ej = np.array([[segs[j][0], segs[j][1]], [segs[j][2], segs[j][3]]], float)
            if np.linalg.norm(ei[:, None, :] - ej[None, :, :], axis=2).min() < dist_th:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def edge_chains(gray, min_span: float = 150.0):
    """从灰度图里提取"可能是同一条物理直线"的点集。"""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    hl = cv2.HoughLinesP(edges, 1, np.pi / 720, 40, minLineLength=35, maxLineGap=12)
    if hl is None:
        return []
    segs = [s[0] for s in hl]
    chains = []
    for idxs in _link_segments(segs):
        pts = []
        for i in idxs:
            s = segs[i]
            pts += [[s[0], s[1]], [s[2], s[3]]]
        pts = np.array(pts, np.float64)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        t = (pts - c) @ vt[0]
        if t.max() - t.min() >= min_span:
            chains.append(pts)
    return chains


def undistort_points(pts, K, dist) -> np.ndarray:
    return cv2.undistortPoints(
        np.asarray(pts, np.float64).reshape(-1, 1, 2), K,
        np.asarray(dist, np.float64).reshape(-1, 1), P=K).reshape(-1, 2)


def chain_bow(pts_undist):
    """返回(弓高, 跨度)：弓高就是这条折线偏离直线的最大二次偏离量(px)。"""
    pts = np.asarray(pts_undist, np.float64)
    if len(pts) < 4:
        return None
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    t = (pts - c) @ vt[0]
    dev = (pts - c) @ vt[1]
    span = float(t.max() - t.min())
    if span < 60:
        return None
    A = np.stack([t ** 2, t, np.ones_like(t)], 1)
    coef, *_ = np.linalg.lstsq(A, dev, rcond=None)
    bow = abs(float(coef[0])) * span ** 2 / 4.0
    return bow, span


def straightness_score(chains, K, dist):
    """把所有链去畸变后量"有多直"，用归一化弓高（越小越直）。"""
    total, n = 0.0, 0
    for pts in chains:
        r = chain_bow(undistort_points(pts, K, dist))
        if r is None:
            continue
        total += r[0] / r[1]
        n += 1
    return (total / n if n else 1e9), n


def collect_chains(frames_gray, min_span: float = 150.0):
    chains = []
    for g in frames_gray:
        chains.extend(edge_chains(g, min_span))
    return chains


def estimate_distortion(chains, frame_size, f: float,
                        k1_lo: float = -0.60, k1_hi: float = 0.20,
                        fit_k2: bool = True, progress=None):
    """固定 f（由先验视场角换算），网格搜索 k1（可选 k2）让所有长链最直。

    注意：f 与 k1 在像素空间是耦合的（只有 c=k1/f² 可辨识），所以这里
    固定 f 再解 k1，效果与任意 f 下的最优解完全等价。
    """
    w, h = int(frame_size[0]), int(frame_size[1])
    K = camera_matrix_from(f, w, h)

    best = (1e9, 0.0)
    for k1 in np.arange(k1_lo, k1_hi + 1e-9, 0.01):
        sc, n = straightness_score(chains, K, dist_vector(k1))
        if n and sc < best[0]:
            best = (sc, float(k1))
        if progress:
            progress()
    # 细化
    lo, hi = best[1] - 0.02, best[1] + 0.02
    for k1 in np.arange(lo, hi + 1e-9, 0.002):
        sc, n = straightness_score(chains, K, dist_vector(k1))
        if n and sc < best[0]:
            best = (sc, float(k1))
        if progress:
            progress()

    k1 = best[1]
    k2 = 0.0
    if fit_k2:
        base = best[0]
        best_k2, best_s = 0.0, base
        for cand in np.arange(-0.25, 0.2501, 0.01):
            sc, n = straightness_score(chains, K, dist_vector(k1, cand))
            if n and sc < best_s * 0.995:
                best_s, best_k2 = sc, float(cand)
            if progress:
                progress()
        k2 = best_k2

    score, n = straightness_score(chains, K, dist_vector(k1, k2))
    return {
        "f": float(f), "k1": k1, "k2": k2,
        "score": float(score), "chains": int(n),
        "c": k1 / (f * f),
        "camera_matrix": K, "dist_coeffs": dist_vector(k1, k2),
    }


def prepare_params(profile, frame_size, strength: float = 1.0):
    """把档案里的 K/畸变系数换算到实际采集尺寸，并套上矫正强度。"""
    K = np.asarray(profile["camera_matrix"], np.float64).reshape(3, 3).copy()
    d = np.asarray(profile["dist_coeffs"], np.float64).reshape(-1).copy()
    pw, ph = profile["image_size"]
    fw, fh = int(frame_size[0]), int(frame_size[1])
    if (int(pw), int(ph)) != (fw, fh):
        sx, sy = fw / float(pw), fh / float(ph)
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
    return K, d * float(strength)


def maps_for(profile, frame_size, crop_size, strength: float = 1.0, max_zoom: float = 1.8):
    """去畸变 + 中心正方形裁切，一步到位（输出就是裁切后的正方形）。"""
    K, d = prepare_params(profile, frame_size, strength)
    mx, my, zoom, bad = build_crop_maps_autozoom(K, d, frame_size, crop_size, max_zoom=max_zoom)
    return mx, my, zoom, bad


def full_maps_for(profile, frame_size, strength: float = 1.0):
    """整幅去畸变映射（用于 pad 模式，或需要完整画面时）。"""
    K, d = prepare_params(profile, frame_size, strength)
    fw, fh = int(frame_size[0]), int(frame_size[1])
    mx, my = cv2.initUndistortRectifyMap(
        K, d.reshape(-1, 1), None, K, (fw, fh), cv2.CV_32FC1)
    return mx, my
