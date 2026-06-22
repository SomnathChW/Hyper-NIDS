"""
Embedding network for Euclidean and Poincaré prototypical networks.

The MLP backbone is **byte-for-byte identical** between methods.
Only the final projection step and prototype registration differ,
ensuring a fair ablation.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List

from modules.poincare_math import exp_map_zero, project_to_ball
from modules.prototypes import (
    generate_euclidean_prototypes,
    generate_orthogonal_prototypes,
)


class EmbeddingNetwork(nn.Module):
    """
    Shared MLP backbone with method-specific output projection and prototypes.

    Architecture::

        Input → [Linear → BN → ReLU → Dropout] × N → Linear(embedding_dim)

    **Euclidean mode:**
        - Output: raw coordinates, optionally L2-normalised.
        - Prototypes: ``nn.Parameter(randn(C, D))`` — **learnable** via
          gradient descent.

    **Poincaré mode:**
        - Output: ``exp_map_zero(v) → project_to_ball(x)``.
        - Prototypes: ``register_buffer`` — **frozen** orthogonal boundary
          points.

    Args:
        input_dim: Number of input features.
        num_classes: Number of known classes (for prototype allocation).
        config: Dictionary containing architecture and method settings.
            Required keys: ``method``, ``hidden_layers``, ``embedding_dim``.
            Optional: ``activation``, ``dropout_rate``, ``curvature``,
            ``clip_radius``, ``prototype_placement_radius``,
            ``embedding_l2_normalize``.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.method: str = config["method"]
        hidden_layers: List[int] = config["hidden_layers"]
        self.embedding_dim: int = config["embedding_dim"]
        activation: str = config.get("activation", "relu")
        dropout_rate: float = config.get("dropout_rate", 0.0)

        # Poincaré parameters
        self.curvature: float = config.get("curvature", 1.0)
        self.clip_radius: float = config.get("clip_radius", 0.95)
        self.max_tangent_norm: float = config.get("max_tangent_norm", 1.0)

        # Euclidean parameters
        self.l2_normalize: bool = config.get("embedding_l2_normalize", False)

        # ── Build shared backbone ────────────────────────────────────
        activation_fn = _get_activation(activation)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for i, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = units

        # Final embedding layer — NO activation (raw coordinates)
        layers.append(nn.Linear(prev_dim, self.embedding_dim))

        self.backbone = nn.Sequential(*layers)

        # ── Method-specific prototype registration ───────────────────
        if self.method == "euclidean":
            # Learnable prototypes — gradient descent optimises placement
            self.prototypes = nn.Parameter(
                generate_euclidean_prototypes(num_classes, self.embedding_dim)
            )
        elif self.method == "poincare":
            # Frozen boundary prototypes — no gradients
            placement_radius = config.get("prototype_placement_radius", 0.95)
            self.register_buffer(
                "prototypes",
                generate_orthogonal_prototypes(
                    num_classes, self.embedding_dim, placement_radius,
                ),
            )
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Expected 'euclidean' or 'poincare'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input features → embedding space.

        Args:
            x: Input features ``[B, input_dim]``.

        Returns:
            Embeddings ``[B, embedding_dim]`` in the appropriate space.
        """
        # Shared backbone
        emb = self.backbone(x)

        if self.method == "euclidean":
            if self.l2_normalize:
                emb = nn.functional.normalize(emb, p=2, dim=1)
        elif self.method == "poincare":
            # Scale backbone output to prevent tanh saturation and dead gradients.
            # A hard clamp kills gradients if the MLP output is large (derivative=0).
            # Scaling by 0.1 keeps inputs in the linear regime of tanh initially,
            # allowing the optimizer to smoothly learn the correct radius.
            emb = emb * 0.1

            # Map tangent vector at origin into the Poincaré ball
            emb = exp_map_zero(emb, c=self.curvature)
            emb = project_to_ball(emb, c=self.curvature, eps=1e-5)

        return emb


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_activation(name: str):
    """Return an activation *class* (not instance) by name."""
    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    name_lower = name.strip().lower()
    if name_lower not in activations:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Choose from: {sorted(activations)}"
        )
    return activations[name_lower]
