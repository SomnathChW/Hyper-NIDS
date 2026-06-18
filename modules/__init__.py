"""
PoincaréBall: Hyperbolic Prototypical Networks for Open-Set NIDS.

Modules:
    poincare_math   - Poincaré ball manifold operations
    prototypes      - Static orthogonal prototype initialisation
    model           - EmbeddingNetwork (Euclidean / Poincaré)
    loss            - Prototypical + SupCon + Center loss functions
    train           - Training loop
    test            - Testing / evaluation with OSR metrics
    utils/          - Data loading, preprocessing, checkpoint I/O
"""
