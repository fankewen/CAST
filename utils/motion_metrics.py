"""TopoCap metrics for comparing predicted motion with a reference."""

import numpy as np
from scipy.spatial.distance import cdist


def topocap_protocol_metrics(pred_position, gt_position, rest_position):
    """Compute MPJPE, PA-MPJPE, MPJVE, and joint-to-joint Chamfer distance."""
    pred = np.asarray(pred_position, dtype=np.float64)
    gt = np.asarray(gt_position, dtype=np.float64)
    rest = np.asarray(rest_position, dtype=np.float64)
    scale = 2.0 / np.ptp(rest, axis=0).max()
    millimeters = 500.0 * scale

    error = pred - gt
    mpjpe = np.linalg.norm(error, axis=-1).mean() * millimeters
    mpjve = (
        np.linalg.norm(np.diff(error, axis=0), axis=-1).mean() * millimeters
        if len(error) > 1 else 0.0
    )

    chamfer = 0.0
    for pred_frame, gt_frame in zip(pred, gt):
        distances = cdist(pred_frame, gt_frame)
        chamfer += distances.min(axis=1).mean() + distances.min(axis=0).mean()
    chamfer *= 0.25 * scale / len(pred)

    aligned_errors = []
    for pred_frame, gt_frame in zip(pred, gt):
        pred_center = pred_frame.mean(axis=0)
        gt_center = gt_frame.mean(axis=0)
        pred_zero = pred_frame - pred_center
        gt_zero = gt_frame - gt_center
        pred_energy = float(np.square(pred_zero).sum())
        if pred_energy <= 1e-12:
            aligned = np.broadcast_to(gt_center, pred_frame.shape)
        else:
            u, singular, vt = np.linalg.svd(pred_zero.T @ gt_zero, full_matrices=False)
            correction = np.ones(3, dtype=np.float64)
            if np.linalg.det(u @ vt) < 0.0:
                correction[-1] = -1.0
            rotation = (u * correction[None, :]) @ vt
            aligned_scale = float((singular * correction).sum()) / pred_energy
            aligned = aligned_scale * (pred_zero @ rotation) + gt_center
        aligned_errors.append(np.linalg.norm(aligned - gt_frame, axis=-1).mean())

    return {
        "topocap_mpjpe_mm": float(mpjpe),
        "topocap_pa_mpjpe_mm": float(np.mean(aligned_errors) * millimeters),
        "topocap_mpjve_mm": float(mpjve),
        "topocap_cd": float(chamfer),
    }
