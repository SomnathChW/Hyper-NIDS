"""
Poincaré Ball manifold operations.

All functions operate on the Poincaré ball model of hyperbolic space with
curvature **-c** (c > 0).  The ball has radius 1/√c.

CRITICAL: Every call to ``torch.atanh`` in this module goes through
:func:`safe_arctanh` which clamps inputs to ``(-1+ε, 1-ε)``.  If a
tensor norm drifts to exactly 1.0 before entering ``arctanh``, the
gradient becomes ``inf`` and training is irrecoverably corrupted.
"""

import torch
from torch import Tensor

# ── Numerical safety constants ───────────────────────────────────────────────
_EPS_NORM = 1e-15    # minimum value for norms (prevent 0-division)
_EPS_ATANH = 1e-7    # clamping margin for arctanh domain


# ── Safe arctanh ─────────────────────────────────────────────────────────────


def safe_arctanh(x: Tensor, eps: float = _EPS_ATANH) -> Tensor:
    """
    Numerically safe ``arctanh``.

    Clamps *x* to the open interval ``(-1 + eps, 1 - eps)`` **before**
    calling ``torch.atanh``, preventing infinite gradients at ±1.

    Args:
        x: Input tensor (any shape).
        eps: Safety margin from ±1.

    Returns:
        ``atanh(clamp(x))`` with the same shape as *x*.
    """
    return torch.atanh(torch.clamp(x, min=-1.0 + eps, max=1.0 - eps))


# ── Conformal factor ─────────────────────────────────────────────────────────


def lambda_x(x: Tensor, c: float = 1.0) -> Tensor:
    """
    Conformal factor λ_x^c = 2 / (1 - c‖x‖²).

    Args:
        x: Points inside the Poincaré ball ``[..., D]``.
        c: Absolute curvature (positive scalar).

    Returns:
        Conformal factor ``[...]`` (last dim reduced).
    """
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
    return 2.0 / torch.clamp(1.0 - c * x_sqnorm, min=_EPS_NORM)


# ── Möbius addition ──────────────────────────────────────────────────────────


def mobius_add(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """
    Möbius addition  x ⊕_c y  in the Poincaré ball.

    Formula::

        x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²) x + (1 - c‖x‖²) y)
                   / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)

    Fully vectorised — supports arbitrary batch dimensions via
    broadcasting on ``[..., D]`` tensors.

    Args:
        x: First operand  ``[..., D]``.
        y: Second operand ``[..., D]``.
        c: Absolute curvature.

    Returns:
        Result ``[..., D]``, projected back onto the ball for safety.
    """
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)  # [..., 1]
    y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

    numerator = (
        (1.0 + 2.0 * c * xy_inner + c * y_sqnorm) * x
        + (1.0 - c * x_sqnorm) * y
    )
    denominator = 1.0 + 2.0 * c * xy_inner + c * c * x_sqnorm * y_sqnorm
    result = numerator / torch.clamp(denominator, min=_EPS_NORM)

    return project_to_ball(result, c)


# ── Exponential map at the origin ────────────────────────────────────────────


def exp_map_zero(v: Tensor, c: float = 1.0) -> Tensor:
    """
    Exponential map at the origin of the Poincaré ball.

    Maps a tangent vector *v* ∈ T_0 M  into the ball::

        exp_0^c(v) = tanh(√c ‖v‖) · v / (√c ‖v‖)

    Args:
        v: Tangent vector at origin ``[..., D]``.
        c: Absolute curvature.

    Returns:
        Point inside the Poincaré ball ``[..., D]``.
    """
    sqrt_c = c ** 0.5
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    v_norm = torch.clamp(v_norm, min=_EPS_NORM)

    result = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return project_to_ball(result, c)


# ── Logarithmic map at the origin ────────────────────────────────────────────


def log_map_zero(y: Tensor, c: float = 1.0) -> Tensor:
    """
    Logarithmic map at the origin (inverse of :func:`exp_map_zero`).

    Maps a point *y* inside the ball back to the tangent space at origin::

        log_0^c(y) = arctanh(√c ‖y‖) · y / (√c ‖y‖)

    Uses :func:`safe_arctanh` internally.

    Args:
        y: Point inside the Poincaré ball ``[..., D]``.
        c: Absolute curvature.

    Returns:
        Tangent vector at origin ``[..., D]``.
    """
    sqrt_c = c ** 0.5
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
    y_norm = torch.clamp(y_norm, min=_EPS_NORM)

    return safe_arctanh(sqrt_c * y_norm) * y / (sqrt_c * y_norm)


# ── Poincaré distance ────────────────────────────────────────────────────────


def poincare_distance(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """
    Geodesic distance on the Poincaré ball.

    Formula::

        d_c(x, y) = (2 / √c) · arctanh(√c ‖ -x ⊕_c y ‖)

    Uses :func:`safe_arctanh` internally via :func:`mobius_add`.

    This function is fully batched.  To compute pairwise distances
    between embeddings ``[B, D]`` and prototypes ``[C, D]``, broadcast
    as ``poincare_distance(emb[:, None, :], proto[None, :, :], c)``
    which yields ``[B, C]``.

    Args:
        x: First set of points  ``[..., D]``.
        y: Second set of points ``[..., D]``.
        c: Absolute curvature.

    Returns:
        Distances ``[...]`` (last dim reduced).
    """
    sqrt_c = c ** 0.5

    # -x ⊕_c y
    neg_x = -x
    add_result = mobius_add(neg_x, y, c)

    add_norm = torch.norm(add_result, p=2, dim=-1)
    add_norm = torch.clamp(add_norm, min=_EPS_NORM)

    dist = (2.0 / sqrt_c) * safe_arctanh(sqrt_c * add_norm)
    return dist


# ── Projection onto the ball ─────────────────────────────────────────────────


def project_to_ball(
    x: Tensor,
    c: float = 1.0,
    eps: float = 1e-5,
) -> Tensor:
    """
    Project points onto the open Poincaré ball.

    Any point whose norm ≥ 1/√c - eps is rescaled to have norm exactly
    ``1/√c - eps``, keeping its direction.

    Args:
        x: Points ``[..., D]``.
        c: Absolute curvature.
        eps: Safety margin from the boundary.

    Returns:
        Projected points ``[..., D]``.
    """
    max_norm = (1.0 / (c ** 0.5)) - eps
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_norm = torch.clamp(x_norm, min=_EPS_NORM)
    cond = x_norm > max_norm
    projected = x / x_norm * max_norm
    return torch.where(cond, projected, x)


# ── Distance from origin ────────────────────────────────────────────────────


def origin_distance(x: Tensor, c: float = 1.0) -> Tensor:
    """
    Distance from each point to the origin of the Poincaré ball.

    Simplified formula (origin has zero norm)::

        d_c(0, x) = (2 / √c) · arctanh(√c ‖x‖)

    Used for the "center void" unknown-detection criterion.

    Args:
        x: Points inside the ball ``[..., D]``.
        c: Absolute curvature.

    Returns:
        Distances ``[...]``.
    """
    sqrt_c = c ** 0.5
    x_norm = torch.norm(x, p=2, dim=-1)
    x_norm = torch.clamp(x_norm, min=_EPS_NORM)

    return (2.0 / sqrt_c) * safe_arctanh(sqrt_c * x_norm)


# ── Fréchet mean (hyperbolic centroid) ───────────────────────────────────────


def poincare_centroid(points: Tensor, c: float = 1.0) -> Tensor:
    """
    Approximate Fréchet mean of points on the Poincaré ball.

    The Fréchet mean minimises the sum of squared geodesic distances
    to all input points — the Riemannian generalisation of the
    arithmetic mean.

    Algorithm:
        1. ``log_map_zero(x_i)``  → tangent vectors at origin
        2. Euclidean mean in tangent space
        3. ``exp_map_zero(v̄)``   → back to the ball

    This is exact when points are clustered and a good first-order
    approximation otherwise.

    Args:
        points: Batch of points ``[N, D]`` inside the Poincaré ball.
        c: Absolute curvature.

    Returns:
        Centroid ``[D]`` inside the Poincaré ball.
    """
    tangent_vectors = log_map_zero(points, c=c)       # [N, D]
    mean_tangent = tangent_vectors.mean(dim=0)         # [D]
    centroid = exp_map_zero(mean_tangent.unsqueeze(0), c=c)  # [1, D]
    return centroid.squeeze(0)                         # [D]
