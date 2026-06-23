"""
Testing / evaluation for Open-Set Recognition.

Loads a trained checkpoint, generates embeddings on test data, classifies
using prototype distances + thresholds, and reports OSR-specific metrics.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from modules.model import EmbeddingNetwork
from modules.poincare_math import origin_distance, poincare_distance
from modules.utils.data_utils import (
    configure_columns_from_dict,
    load_data,
    preprocess_data,
    split_features_and_target,
)
from modules.utils.model_utils import load_checkpoint


# ── Public API ───────────────────────────────────────────────────────────────


def run_testing(method: str, test_data_path: str) -> Dict[str, Any]:
    """
    Evaluate a trained model with Open-Set Recognition metrics.

    Pipeline:
        1. Load checkpoint (model, scaler, prototypes, thresholds)
        2. Load & preprocess test data
        3. Generate embeddings
        4. Classify: nearest prototype + threshold → "Unknown" detection
        5. Compute & report OSR metrics
        6. Save result plots

    Args:
        method: The method to test ("euclidean" or "poincare").
        test_data_path: Path to the test data.

    Returns:
        Dictionary with predictions, distances, metrics, and paths.
    """
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"\n[bold]Device:[/bold] {device}")

    # ── [1] Load checkpoint ──────────────────────────────────────────
    console.print("\n[bold cyan][1/5] Loading checkpoint...[/bold cyan]")
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models"
    results_dir = project_root / "results" / method
    results_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_checkpoint(model_dir, method_name=method)
    state_dict = artifacts["state_dict"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    column_config = artifacts["column_config"]
    model_config = artifacts["model_config"]
    thresholds = artifacts.get("thresholds", {})
    per_class_thresholds = artifacts.get("per_class_thresholds", {})

    num_classes = len(label_encoder.classes_)

    # Rebuild model and load weights
    # Infer input_dim from state dict
    first_weight_key = [k for k in state_dict if "backbone.0.weight" in k][0]
    input_dim = state_dict[first_weight_key].shape[1]

    model = EmbeddingNetwork(input_dim, num_classes, model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    console.print(f"  ✓ Model loaded ({method}, {num_classes} classes)")

    # ── [2] Load test data ───────────────────────────────────────────
    console.print("\n[bold cyan][2/5] Loading test data...[/bold cyan]")
    if not test_data_path:
        console.print("[red]ERROR: No test data path specified.[/red]")
        return {"error": "No test data path"}

    max_samples_per_file = model_config.get("max_samples_per_file", 100000)
    df_test = load_data(
        file_path=test_data_path,
        max_samples_per_file=max_samples_per_file,
    )

    # ── [3] Preprocess ───────────────────────────────────────────────
    console.print("\n[bold cyan][3/5] Preprocessing...[/bold cyan]")
    X_test, y_test = split_features_and_target(df_test, column_config)
    X_test = preprocess_data(X_test, handle_inf=True, handle_nan=True, fill_value=0)
    X_test_scaled = scaler.transform(X_test)
    del X_test

    # ── [4] Classify ─────────────────────────────────────────────────
    console.print("\n[bold cyan][4/5] Generating embeddings & classifying...[/bold cyan]")
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
    del X_test_scaled

    with torch.no_grad():
        embeddings_list = []
        chunk_size = 4096
        for i in range(0, X_test_t.shape[0], chunk_size):
            embeddings_list.append(model(X_test_t[i:i+chunk_size]))
        embeddings = torch.cat(embeddings_list, dim=0)
        prototypes = model.prototypes

        predictions, min_distances, is_unknown = _classify(
            embeddings, prototypes, method, model_config,
            thresholds, per_class_thresholds, label_encoder,
        )

    del X_test_t

    # ── [5] Metrics ──────────────────────────────────────────────────
    console.print("\n[bold cyan][5/5] Computing metrics...[/bold cyan]")

    # True labels (string form)
    target_col = column_config.get("target_column")
    y_true = None
    test_metrics = None

    if y_test is not None and target_col:
        y_true = y_test.astype(str).to_numpy()
        known_labels = [str(c) for c in label_encoder.classes_]
        
        # Collapse all novel/held-out classes into "Unknown" for standard OSR evaluation
        is_truly_known_mask = np.isin(y_true, known_labels)
        y_true[~is_truly_known_mask] = "Unknown"

        test_metrics = _compute_osr_metrics(
            y_true, predictions, is_unknown, known_labels,
        )

        _print_metrics(console, test_metrics, method)

    # ── Save plots ───────────────────────────────────────────────────
    _save_test_plots(
        embeddings.cpu().numpy(),
        predictions,
        min_distances,
        is_unknown,
        y_true,
        label_encoder,
        results_dir,
        method,
        curvature=model_config.get("curvature", 1.0),
    )

    # Cleanup
    del embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.print("\n" + "=" * 80)
    console.print("[bold green]TESTING COMPLETED![/bold green]")
    console.print("=" * 80)

    return {
        "predictions": predictions,
        "min_distances": min_distances,
        "is_unknown": is_unknown,
        "metrics": test_metrics,
        "results_dir": str(results_dir),
    }


# ── Classification ───────────────────────────────────────────────────────────


def _classify(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    method: str,
    config: Dict[str, Any],
    thresholds: Dict[str, float],
    per_class_thresholds: Dict[int, float],
    label_encoder,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify embeddings by nearest prototype + threshold-based unknown detection.

    Returns:
        (predictions, min_distances, is_unknown) — all numpy arrays.
    """
    # Compute distances to all prototypes [N, C]
    if method == "euclidean":
        diff = embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=2)
        metric = config.get("distance_metric", "squared_euclidean")
        if metric == "euclidean":
            distances = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
        else:
            distances = dist_sq
    else:
        distances = poincare_distance(
            embeddings.unsqueeze(1),
            prototypes.unsqueeze(0),
            c=config.get("curvature", 1.0),
        )

    # Nearest prototype
    min_dists, nearest_idx = distances.min(dim=1)
    min_dists_np = min_dists.cpu().numpy()
    nearest_idx_np = nearest_idx.cpu().numpy()

    # Map indices to class names
    predictions = np.array([
        str(label_encoder.classes_[idx]) for idx in nearest_idx_np
    ], dtype=object)

    # Unknown detection: per-class threshold
    is_unknown = np.zeros(len(predictions), dtype=bool)
    for i, (cls_idx, dist_val) in enumerate(zip(nearest_idx_np, min_dists_np)):
        tau_k = per_class_thresholds.get(int(cls_idx), thresholds.get("global", float("inf")))
        if dist_val > tau_k:
            is_unknown[i] = True

    # Poincaré: additional origin-distance criterion
    if method == "poincare" and "origin" in thresholds:
        origin_dists = origin_distance(
            embeddings, c=config.get("curvature", 1.0),
        ).cpu().numpy()
        origin_threshold = thresholds["origin"]
        # Samples too close to origin → unknown
        origin_unknown = origin_dists < origin_threshold
        is_unknown = is_unknown | origin_unknown

    # Mark unknowns
    predictions[is_unknown] = "Unknown"

    return predictions, min_dists_np, is_unknown


# ── OSR Metrics ──────────────────────────────────────────────────────────────


def _compute_osr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_unknown_pred: np.ndarray,
    known_labels: List[str],
) -> Dict[str, Any]:
    """
    Compute Open-Set Recognition metrics.

    Metrics:
        - Known-class accuracy (on truly-known samples predicted as known)
        - Unknown Detection Rate (UDR): recall on unknown samples
        - False Positive Rate (FPR): known samples wrongly flagged as unknown
        - Open-Set F1: harmonic mean of UDR and (1 - FPR)
        - Per-class precision, recall, F1
        - Confusion matrix (with Unknown row/column)
    """
    known_set = set(known_labels)

    # Partition ground truth
    is_truly_known = np.array([y in known_set for y in y_true])
    is_truly_unknown = ~is_truly_known

    metrics: Dict[str, Any] = {
        "total_samples": len(y_true),
        "known_samples": int(is_truly_known.sum()),
        "unknown_samples": int(is_truly_unknown.sum()),
    }

    # ── Known-class accuracy ─────────────────────────────────────────
    # Among truly-known samples that are predicted as known (not "Unknown")
    known_mask = is_truly_known & ~is_unknown_pred
    if known_mask.any():
        metrics["known_class_accuracy"] = float(
            accuracy_score(y_true[known_mask], y_pred[known_mask])
        )
    else:
        metrics["known_class_accuracy"] = 0.0

    # ── Unknown Detection Rate (UDR) ─────────────────────────────────
    if is_truly_unknown.any():
        metrics["unknown_detection_rate"] = float(
            is_unknown_pred[is_truly_unknown].mean()
        )
    else:
        metrics["unknown_detection_rate"] = float("nan")

    # ── False Positive Rate (FPR) ────────────────────────────────────
    if is_truly_known.any():
        metrics["false_positive_rate"] = float(
            is_unknown_pred[is_truly_known].mean()
        )
    else:
        metrics["false_positive_rate"] = float("nan")

    # ── Open-Set F1 ──────────────────────────────────────────────────
    udr = metrics["unknown_detection_rate"]
    fpr = metrics["false_positive_rate"]
    if not (np.isnan(udr) or np.isnan(fpr)):
        precision_unk = 1.0 - fpr if (1.0 - fpr) > 0 else 0.0
        if udr + precision_unk > 0:
            metrics["open_set_f1"] = float(
                2 * udr * precision_unk / (udr + precision_unk)
            )
        else:
            metrics["open_set_f1"] = 0.0
    else:
        metrics["open_set_f1"] = float("nan")

    # ── Weighted F1 across all classes (including Unknown) ───────────
    all_labels = sorted(known_labels) + ["Unknown"]
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)
    )
    metrics["weighted_precision"] = float(
        precision_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)
    )
    metrics["weighted_recall"] = float(
        recall_score(y_true, y_pred, labels=all_labels, average="weighted", zero_division=0)
    )

    # ── Confusion matrix ─────────────────────────────────────────────
    unique_labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_labels"] = unique_labels

    return metrics


# ── Printing ─────────────────────────────────────────────────────────────────


def _print_metrics(
    console: Console,
    metrics: Dict[str, Any],
    method: str,
) -> None:
    """Print OSR metrics as a Rich table."""
    table = Table(
        title=f"Open-Set Recognition Results ({method.upper()})",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white", justify="right")

    table.add_row("Total samples", str(metrics["total_samples"]))
    table.add_row("Known samples", str(metrics["known_samples"]))
    table.add_row("Unknown samples", str(metrics["unknown_samples"]))
    table.add_row("", "")  # separator
    table.add_row(
        "Known-Class Accuracy",
        f"{metrics['known_class_accuracy']:.4f}",
    )
    table.add_row(
        "Unknown Detection Rate (UDR)",
        f"{metrics['unknown_detection_rate']:.4f}",
    )
    table.add_row(
        "False Positive Rate (FPR)",
        f"{metrics['false_positive_rate']:.4f}",
    )
    table.add_row(
        "Open-Set F1",
        f"{metrics['open_set_f1']:.4f}",
    )
    table.add_row("", "")
    table.add_row("Weighted Precision", f"{metrics['weighted_precision']:.4f}")
    table.add_row("Weighted Recall", f"{metrics['weighted_recall']:.4f}")
    table.add_row("Weighted F1", f"{metrics['weighted_f1']:.4f}")

    console.print(table)


# ── Plotting ─────────────────────────────────────────────────────────────────


def _save_test_plots(
    embeddings: np.ndarray,
    predictions: np.ndarray,
    min_distances: np.ndarray,
    is_unknown: np.ndarray,
    y_true: Optional[np.ndarray],
    label_encoder,
    results_dir: Path,
    method: str,
    curvature: float = 1.0,
) -> None:
    """Save confusion matrix and distance distribution plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Use true labels if available, otherwise fallback to predictions
        if y_true is not None:
            plot_mask_unknown = (y_true == "Unknown")
        else:
            plot_mask_unknown = is_unknown

        # ── Confusion matrix ─────────────────────────────────────────
        if y_true is not None:
            unique = sorted(set(y_true.tolist() + predictions.tolist()))
            cm = confusion_matrix(y_true, predictions, labels=unique)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique, yticklabels=unique, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix ({method.upper()})")
            path = results_dir / "confusion_matrix.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ Saved confusion matrix → {path}")

        # ── Distance distribution ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 5))
        
        known_dists = min_distances[~plot_mask_unknown]
        unknown_dists = min_distances[plot_mask_unknown]

        if len(known_dists) > 0:
            ax.hist(known_dists, bins=50, alpha=0.6, label="Known", color="steelblue")
        if len(unknown_dists) > 0:
            ax.hist(unknown_dists, bins=50, alpha=0.6, label="Unknown", color="tomato")
        ax.set_xlabel("Distance to Nearest Prototype")
        ax.set_ylabel("Count")
        ax.set_title(f"Distance Distribution ({method.upper()})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = results_dir / "distance_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved distance distribution → {path}")

        # ── 2D Projection (UMAP/PCA) ─────────────────────────────────
        try:
            if method == "poincare":
                # Poincaré: compute N×N geodesic distance matrix and
                # use UMAP with metric='precomputed'.  Standard cosine
                # or euclidean metrics are geometrically invalid here.
                max_points = 8000  # N×N matrix must fit in memory
                if len(embeddings) > max_points:
                    idx = np.random.choice(len(embeddings), max_points, replace=False)
                    plot_emb = embeddings[idx]
                    plot_is_unknown = plot_mask_unknown[idx]
                else:
                    plot_emb = embeddings
                    plot_is_unknown = plot_mask_unknown

                plot_tensor = torch.from_numpy(plot_emb).float()
                n = len(plot_tensor)
                dist_matrix = np.zeros((n, n), dtype=np.float32)
                chunk = 256
                for i in range(0, n, chunk):
                    ei = min(i + chunk, n)
                    row_dists = poincare_distance(
                        plot_tensor[i:ei].unsqueeze(1),
                        plot_tensor.unsqueeze(0),
                        c=curvature,
                    ).numpy()
                    dist_matrix[i:ei] = row_dists

                try:
                    import umap
                    reducer = umap.UMAP(
                        metric="precomputed", n_components=2, random_state=42,
                    )
                    algo_name = "UMAP (Poincaré precomputed)"
                except ImportError:
                    from sklearn.manifold import MDS
                    reducer = MDS(
                        n_components=2, dissimilarity="precomputed",
                        random_state=42, normalized_stress="auto",
                    )
                    algo_name = "MDS (Poincaré precomputed)"

                emb_2d = reducer.fit_transform(dist_matrix)

                # Scale into unit disk for visual consistency
                max_r = np.max(np.linalg.norm(emb_2d, axis=1))
                if max_r > 1e-12:
                    emb_2d = emb_2d / max_r * 0.95

            else:
                # Euclidean: standard UMAP / PCA
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
                    algo_name = "UMAP"
                except ImportError:
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2)
                    algo_name = "PCA"

                max_points = 20000 if algo_name == "UMAP" else 50000
                if len(embeddings) > max_points:
                    idx = np.random.choice(len(embeddings), max_points, replace=False)
                    plot_emb = embeddings[idx]
                    plot_is_unknown = plot_mask_unknown[idx]
                else:
                    plot_emb = embeddings
                    plot_is_unknown = plot_mask_unknown

                emb_2d = reducer.fit_transform(plot_emb)

            fig, ax = plt.subplots(figsize=(8, 8))

            # Draw unit circle if Poincare
            if method == "poincare":
                circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', alpha=0.5)
                ax.add_patch(circle)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)

            # Scatter Knowns
            if len(emb_2d[~plot_is_unknown]) > 0:
                ax.scatter(
                    emb_2d[~plot_is_unknown, 0], emb_2d[~plot_is_unknown, 1],
                    c='steelblue', alpha=0.3, s=5, label='Known',
                )
            # Scatter Unknowns
            if len(emb_2d[plot_is_unknown]) > 0:
                ax.scatter(
                    emb_2d[plot_is_unknown, 0], emb_2d[plot_is_unknown, 1],
                    c='tomato', alpha=0.5, s=15, label='Unknown',
                )

            ax.set_title(f"2D Projection ({method.upper()} via {algo_name})")
            ax.legend()
            ax.grid(True, alpha=0.2)

            path = results_dir / f"embedding_space.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ Saved 2D embedding space ({algo_name}) → {path}")

        except Exception as e:
            print(f"  [WARNING] Could not save 2D projection: {e}")

        # ── Norm distribution (Poincaré diagnostic) ──────────────────
        if method == "poincare":
            try:
                emb_norms = np.linalg.norm(embeddings, axis=1)
                known_norms = emb_norms[~plot_mask_unknown]
                unknown_norms = emb_norms[plot_mask_unknown]

                fig, ax = plt.subplots(figsize=(10, 5))
                if len(known_norms) > 0:
                    ax.hist(known_norms, bins=80, alpha=0.6,
                            label="Known", color="steelblue", density=True)
                if len(unknown_norms) > 0:
                    ax.hist(unknown_norms, bins=80, alpha=0.6,
                            label="Unknown", color="tomato", density=True)
                ax.axvline(x=0.7, color="green", linestyle="--",
                           alpha=0.7, label="Target norm (r=0.7)")
                ax.set_xlabel("Euclidean Norm ‖x‖ (radius in ball)")
                ax.set_ylabel("Density")
                ax.set_title("Embedding Norm Distribution (POINCARÉ)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 1.0)

                path = results_dir / "norm_distribution.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  ✓ Saved norm distribution → {path}")
            except Exception as e:
                print(f"  [WARNING] Could not save norm distribution: {e}")

    except Exception as e:
        print(f"  [WARNING] Could not save test plots: {e}")
