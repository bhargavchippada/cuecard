"""Shared numeric utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def l2_normalize(vec: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """L2-normalize vectors, handling both 1D and 2D arrays safely.

    For 1D input, returns a 1D unit vector.
    For 2D input, normalizes each row independently.
    Zero-norm vectors are left as zero (no NaN).
    """
    if vec.ndim == 1:
        norm = np.maximum(np.linalg.norm(vec), 1e-12)
        result: npt.NDArray[np.floating] = (vec / norm).astype(vec.dtype)
        return result
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    result = (vec / norms).astype(vec.dtype)
    return result
