"""Vectorised point-to-segment geometry."""
import numpy as np


def point_segment(p, a, b):
    """Distance from points p to segments a->b (all (...,2) arrays, broadcast).

    Returns (dist, t) where t in [0,1] is the projection position along the
    segment (0 = start node, 1 = end node).
    """
    ab = b - a
    ap = p - a
    L2 = np.einsum("...i,...i->...", ab, ab)
    t = np.where(L2 > 0, np.einsum("...i,...i->...", ap, ab) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1), t


def angle_diff(a, b):
    """Absolute difference between two angles in degrees, in [0,180]."""
    d = np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)
    return d
