"""
Minimalist Embedding Network for Hyperbolic classification.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List

from modules.poincare_math import exp_map_zero
from modules.prototypes import generate_orthogonal_prototypes


class EmbeddingNetwork(nn.Module):
    """
    Minimalist MLP backbone for hyperbolic classification.

    Architecture::
        Input → [Linear → ReLU] × N → Linear(embedding_dim) → exp_map_zero

    - No BatchNorm
    - No Dropout
    - Final layer weights are initialized extremely small (0.01) to prevent
      embeddings from snapping to the boundary of the Poincaré disk.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.method: str = config.get("method", "poincare")
        hidden_layers: List[int] = config["hidden_layers"]
        self.embedding_dim: int = config["embedding_dim"]
        
        # Poincaré parameters
        self.curvature: float = config.get("curvature", 1.0)
        placement_radius = config.get("prototype_placement_radius", 0.8)

        # ── Build shared backbone ────────────────────────────────────
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            prev_dim = units

        # Final embedding layer
        self.final_layer = nn.Linear(prev_dim, self.embedding_dim)
        
        # Explicitly shrink final layer weights
        self.final_layer.weight.data.mul_(0.01)
        self.final_layer.bias.data.zero_()

        self.backbone = nn.Sequential(*layers)

        # ── Method-specific prototype registration ───────────────────
        if self.method == "poincare":
            # Frozen boundary prototypes — no gradients
            self.register_buffer(
                "prototypes",
                generate_orthogonal_prototypes(
                    num_classes, self.embedding_dim, placement_radius,
                ),
            )
        elif self.method == "euclidean":
            # Just in case we need to compare against Euclidean
            self.prototypes = nn.Parameter(
                torch.randn(num_classes, self.embedding_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input features → embedding space.
        """
        # Euclidean Engine
        emb = self.backbone(x)
        emb = self.final_layer(emb)

        if self.method == "poincare":
            # The Bridge: project into the hyperbolic fishbowl
            emb = exp_map_zero(emb, c=self.curvature)

        return emb
