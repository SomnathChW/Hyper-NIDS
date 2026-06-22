"""
Loss functions for Euclidean and Hyperbolic prototypical networks.

All functions are designed for **full GPU execution**:
  • Zero Python ``for`` loops over batch or class dimensions.
  • Pairwise distances via ``(B,1,D) - (1,C,D)`` broadcast → single kernel.
  • SupCon similarity via ``torch.mm`` (one cuBLAS call).
  • Numerical stability via log-sum-exp trick (subtract row-max).
  • ``torch.compile``-friendly: no data-dependent control flow.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.poincare_math import poincare_distance


# ── Supervised Contrastive Loss (shared helper) ──────────────────────────────


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls same-class embeddings together and pushes different-class
    embeddings apart at the pairwise level.

    GPU execution path:
        1. L2-normalise embeddings                              → [B, D]
        2. Cosine similarity: ``mm(emb, emb.T)``               → [B, B]
        3. Positive mask: ``labels[:, None] == labels[None, :]``→ [B, B] bool
        4. Self-mask: ``~eye(B)``
        5. Log-sum-exp with row-max subtraction (stability)
        6. Mean reduction over anchors with ≥1 positive

    Args:
        embeddings: Batch embeddings ``[B, D]``.
        labels: Integer class labels ``[B]``.
        temperature: Sharpness parameter τ_sc.

    Returns:
        Scalar SupCon loss.
    """
    # L2 normalise for cosine similarity
    emb = F.normalize(embeddings, p=2, dim=1)

    batch_size = emb.shape[0]
    device = emb.device

    # Pairwise cosine similarity [B, B]
    sim_matrix = torch.mm(emb, emb.T)

    # Masks — all boolean, GPU-resident
    labels_col = labels.unsqueeze(1)               # [B, 1]
    labels_row = labels.unsqueeze(0)               # [1, B]
    labels_equal = labels_col.eq(labels_row)        # [B, B]
    self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = labels_equal & self_mask         # [B, B] same class, not self
    all_mask = self_mask.float()                     # [B, B] everything except self

    # Scaled logits with numerical stability
    logits = sim_matrix / temperature
    logits_max = logits.max(dim=1, keepdim=True).values
    logits = logits - logits_max.detach()

    # Denominator: sum of exp over all non-self pairs
    exp_logits = torch.exp(logits) * all_mask       # [B, B]
    log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # Log probabilities for positive pairs
    log_probs = logits - log_sum_exp                # [B, B]

    # Mean log-prob over positive pairs per anchor
    positive_mask_f = positive_mask.float()
    num_positives = positive_mask_f.sum(dim=1)      # [B]
    # Avoid division-by-zero for classes with a single sample in batch
    num_positives = torch.clamp(num_positives, min=1.0)

    mean_log_probs = (positive_mask_f * log_probs).sum(dim=1) / num_positives

    # Loss = negative mean log probability
    loss = -mean_log_probs.mean()
    return loss


# ── Euclidean Prototypical Loss ──────────────────────────────────────────────


def euclidean_prototypical_loss(
    embeddings: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    distance_metric: str = "squared_euclidean",
    temperature: float = 0.1,
    center_loss_radius: float = 0.0,
    center_loss_weight: float = 0.1,
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.07,
) -> Tensor:
    """
    Composite Euclidean prototypical loss with toggleable auxiliary terms.

    ``L_total = L_proto + λ_cl·L_center + λ_sc·L_supcon``

    Where:
      - L_proto  = ``-mean(log_softmax(-d / τ)[correct])``
      - L_center = ``mean(ReLU(d²_correct - R²))``  *(if R > 0)*
      - L_supcon = SupCon loss on embeddings          *(if λ_sc > 0)*

    GPU execution:
      • Distances: ``(B,1,D) - (1,C,D)`` broadcast → reduce → ``[B,C]``
      • Correct-class gather: ``torch.gather`` on ``[B,C]`` matrix
      • ReLU penalty: fused element-wise on ``[B]`` tensor
      • All three terms summed as GPU scalars; single ``.backward()``

    Args:
        embeddings: Batch embeddings ``[B, D]``.
        labels: Integer class labels ``[B]``.
        prototypes: Class prototypes ``[C, D]``.
        distance_metric: ``"euclidean"`` or ``"squared_euclidean"``.
        temperature: Softmax scaling τ.
        center_loss_radius: Safe-zone radius *R* (0.0 = disabled).
        center_loss_weight: Scale factor λ_cl.
        supcon_weight: Scale factor λ_sc (0.0 = disabled).
        supcon_temperature: SupCon temperature τ_sc.

    Returns:
        Scalar composite loss.
    """
    # ── Pairwise squared Euclidean distances [B, C] ──────────────────
    # (B,1,D) - (1,C,D) → (B,C,D) → sum → (B,C)
    diff = embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
    distances_sq = (diff * diff).sum(dim=2)

    if distance_metric == "euclidean":
        distances = torch.sqrt(torch.clamp(distances_sq, min=1e-12))
    else:
        distances = distances_sq

    # ── Prototypical loss (negative log softmax) ─────────────────────
    scaled_logits = -distances / temperature
    log_probs = F.log_softmax(scaled_logits, dim=1)

    # Gather log-probs at correct class index
    labels_idx = labels.unsqueeze(1).long()             # [B, 1]
    correct_log_probs = torch.gather(log_probs, 1, labels_idx).squeeze(1)
    loss = -correct_log_probs.mean()

    # ── Bounded Center Loss (optional) ───────────────────────────────
    if center_loss_radius > 0.0:
        correct_dist_sq = torch.gather(
            distances_sq, 1, labels_idx,
        ).squeeze(1)                                     # [B]
        penalty = F.relu(correct_dist_sq - center_loss_radius ** 2)
        loss = loss + center_loss_weight * penalty.mean()

    # ── Supervised Contrastive Loss (optional) ───────────────────────
    if supcon_weight > 0.0:
        loss = loss + supcon_weight * supervised_contrastive_loss(
            embeddings, labels, temperature=supcon_temperature,
        )

    return loss


# ── Hyperbolic Geodesic Pull + Push Loss ─────────────────────────────────────


def hyperbolic_prototypical_loss(
    embeddings: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    curvature: float = 1.0,
    margin: float = 0.5,
    push_margin: float = 1.0,
    push_weight: float = 0.5,
) -> Tensor:
    """
    Geodesic pull + triplet push loss on the Poincaré ball.

    Combines two complementary terms:

    1. **Pull** — ``ReLU(d_correct - margin)``: each sample is attracted
       to its assigned prototype.  Zero loss once within ``margin``.

    2. **Push** — ``ReLU(d_correct - d_wrong_min + push_margin)``: the
       correct prototype must be at least ``push_margin`` geodesic units
       closer than the nearest wrong prototype.  This creates a geometric
       "moat" between class clusters without the softmax
       boundary-inflation problem.

    ``L = L_pull + push_weight · L_push``

    Why both terms are needed
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    Pull-only has zero inter-class signal: nothing prevents class A's
    embeddings from drifting into class B's region.  The push term
    provides a controlled repulsive force that operates on relative
    distances (correct vs. nearest wrong), not absolute softmax
    probabilities, so it does not inflate embeddings toward the boundary.

    GPU execution
    ~~~~~~~~~~~~~
    1. Pairwise Poincaré distances via broadcast         → [B, C]
    2. ``torch.gather`` on correct-class column          → [B]
    3. ``masked_fill`` + ``min`` for nearest wrong       → [B]
    4. Element-wise ``ReLU`` for both terms               → [B]
    5. Scalar ``mean``                                   → []

    No Python loops, ``torch.compile``-friendly.

    Args:
        embeddings: Batch embeddings inside the Poincaré ball ``[B, D]``.
        labels: Integer class labels ``[B]``.
        prototypes: Frozen boundary prototypes ``[C, D]``.
        curvature: Absolute curvature *c* > 0.
        margin: Pull safe-zone radius in geodesic distance units.  Loss
            is zero once an embedding is within ``margin`` of its
            prototype.
        push_margin: Minimum required geodesic gap between d_correct and
            the nearest wrong prototype.  Larger values enforce stronger
            inter-class separation.
        push_weight: Scale factor for the push term (0.0 = disabled).

    Returns:
        Scalar loss.
    """
    # ── Pairwise Poincaré distances [B, C] ───────────────────────────
    distances = poincare_distance(
        embeddings.unsqueeze(1),
        prototypes.unsqueeze(0),
        c=curvature,
    )                                                    # [B, C]

    labels_idx = labels.unsqueeze(1).long()              # [B, 1]

    # ── Pull: distance to correct prototype ──────────────────────────
    d_correct = torch.gather(distances, 1, labels_idx).squeeze(1)  # [B]
    pull_loss = F.relu(d_correct - margin).mean()

    # ── Push: gap from nearest wrong prototype ───────────────────────
    if push_weight > 0.0:
        # Mask out the correct class → find nearest wrong prototype
        wrong_mask = torch.ones_like(distances, dtype=torch.bool)
        wrong_mask.scatter_(1, labels_idx, False)             # [B, C]
        d_wrong_min = distances.masked_fill(
            ~wrong_mask, float("inf"),
        ).min(dim=1).values                                   # [B]

        # Triplet-style: correct must be push_margin closer than wrong
        push_loss = F.relu(d_correct - d_wrong_min + push_margin).mean()

        return pull_loss + push_weight * push_loss

    return pull_loss
