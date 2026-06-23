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

        # ── Reconstruction Decoder ───────────────────────────────────
        self.reconstruction_weight = config.get(self.method, {}).get("reconstruction_weight", 0.0)
        if self.reconstruction_weight > 0.0:
            self.decoder = nn.Sequential(
                nn.Linear(self.embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, input_dim)
            )

        # ── Method-specific prototype registration ───────────────────
        if self.method == "poincare":
            # Start with perfectly spaced unit directions
            ortho_dirs = generate_orthogonal_prototypes(num_classes, self.embedding_dim, placement_radius=1.0)
            self.raw_prototypes = nn.Parameter(ortho_dirs)
            self.placement_radius = placement_radius
        else:
            # Euclidean prototypes start near origin
            self.raw_prototypes = nn.Parameter(
                torch.randn(num_classes, self.embedding_dim) * 0.01
            )

    @property
    def prototypes(self) -> torch.Tensor:
        """
        Fixed radius, learnable angles.
        """
        if self.method == "poincare":
            # 1. Extract purely the direction
            norms = torch.norm(self.raw_prototypes, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            directions = self.raw_prototypes / norms
            # 2. Anchor firmly to the boundary
            return directions * self.placement_radius
        return self.raw_prototypes

    def forward(self, x: torch.Tensor, return_recon: bool = False) -> torch.Tensor:
        """
        Forward pass: input features → embedding space.
        """
        # Euclidean Engine
        emb = self.backbone(x)
        emb = self.final_layer(emb)

        recon = None
        if return_recon and self.reconstruction_weight > 0.0:
            recon = self.decoder(emb)

        if self.method == "poincare":
            # The Bridge: project into the hyperbolic fishbowl
            emb = exp_map_zero(emb, c=self.curvature)

        if return_recon:
            return emb, recon
        return emb
