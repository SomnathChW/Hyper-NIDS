"""
Prototype initialisation for Poincaré and Euclidean embedding spaces.

Poincaré prototypes are **static** (frozen buffer) — maximally separated
orthogonal points placed near the ball boundary.

Euclidean prototypes are **learnable** (``nn.Parameter``) — initialised
randomly and optimised by gradient descent alongside the backbone.
"""

import torch
from torch import Tensor


def generate_orthogonal_prototypes(
    num_classes: int,
    embedding_dim: int,
    placement_radius: float = 0.8,
) -> Tensor:
    """
    Create maximally separated prototype directions, scaled to
    *placement_radius*.

    Strategy:
        1. If ``embedding_dim >= num_classes``: use the first *num_classes*
           standard basis vectors (one-hot), which are perfectly orthogonal.
        2. If ``embedding_dim < num_classes``: draw a random Gaussian matrix
           ``[num_classes, embedding_dim]`` and apply QR decomposition to
           get *embedding_dim* orthonormal rows, then fill the remaining
           rows with the next-best directions from the Q factor.
        3. L2-normalise each row and scale by *placement_radius*.

    Args:
        num_classes: Number of known classes (*C*).
        embedding_dim: Embedding dimensionality (*D*).
        placement_radius: Norm of each prototype (how close to boundary).

    Returns:
        ``Tensor [C, D]`` — prototype coordinates.  Detached, requires
        no gradient.
    """
    if embedding_dim >= num_classes:
        # Perfect one-hot basis — maximally orthogonal
        prototypes = torch.eye(num_classes, embedding_dim)
    else:
        # More classes than dimensions: QR on a random matrix
        # gives the best spread we can achieve
        random_matrix = torch.randn(num_classes, embedding_dim)
        q, _ = torch.linalg.qr(random_matrix.T)  # [D, C] orthonormal cols
        prototypes = q.T[:num_classes]             # [C, D]

    # L2-normalise each prototype and scale to placement_radius
    norms = torch.norm(prototypes, p=2, dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    prototypes = prototypes / norms * placement_radius

    return prototypes.detach()


def generate_euclidean_prototypes(
    num_classes: int,
    embedding_dim: int,
) -> Tensor:
    """
    Create initial prototype positions for the Euclidean baseline.

    These are **random** positions that will be optimised via gradient
    descent as ``nn.Parameter`` tensors.

    Args:
        num_classes: Number of known classes (*C*).
        embedding_dim: Embedding dimensionality (*D*).

    Returns:
        ``Tensor [C, D]`` — initial prototype coordinates (not detached,
        intended to be wrapped in ``nn.Parameter``).
    """
    return torch.randn(num_classes, embedding_dim)
