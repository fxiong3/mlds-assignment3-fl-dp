"""Laplace mechanism for input perturbation (Part 2): X' = X + eps, eps_ij ~ Lap(0, b)."""
from __future__ import annotations

import numpy as np


def laplace_perturb(x: np.ndarray, b: float, rng: np.random.Generator, clip: bool = False) -> np.ndarray:
    """Add i.i.d. zero-mean Laplace(b) noise to every pixel.
    Edge cases: b == 0 returns an exact copy (no RNG draw); b < 0 is rejected; empty input is fine.
    `clip=True` projects back to [0,1] (post-processing, so it does not weaken the DP guarantee)."""
    if b < 0:
        raise ValueError("Laplace scale b must be >= 0")
    x = np.asarray(x, dtype=np.float32)
    if b == 0 or x.size == 0:
        return x.copy()
    out = x + rng.laplace(0.0, b, size=x.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0) if clip else out


def epsilon_per_pixel(b: float, sensitivity: float = 1.0) -> float:
    """Pixels lie in [0,1], so per-pixel L1 sensitivity is 1 and Lap(b) gives eps = 1/b per pixel.
    For a whole 784-pixel image (L1 sensitivity 784) the same noise gives eps = 784/b."""
    return float("inf") if b == 0 else sensitivity / b


def epsilon_per_image(b: float, pixels: int = 784) -> float:
    return epsilon_per_pixel(b, sensitivity=float(pixels))


def snr_db(x: np.ndarray, b: float) -> float:
    """Signal-to-noise ratio of the perturbed images (noise variance = 2 b^2)."""
    if b == 0:
        return float("inf")
    return float(10 * np.log10(np.var(x) / (2 * b * b)))
