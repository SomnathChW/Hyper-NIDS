"""
Training loop for both Euclidean and Poincaré prototypical networks.

Unified pipeline — the ``method`` key in config selects the loss function
and prototype handling.  Everything else (data loading, optimizer, LR
scheduling, early stopping, checkpointing) is shared.
"""

import gc
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from modules.loss import euclidean_prototypical_loss, hyperbolic_prototypical_loss
from modules.model import EmbeddingNetwork
from modules.poincare_math import origin_distance, poincare_distance, project_to_ball
from modules.utils.data_utils import (
    configure_columns_from_dict,
    load_data,
    LogQuantileScaler,
    preprocess_data,
    split_features_and_target,
)
from modules.utils.model_utils import save_checkpoint

# ── Scaler registry ──────────────────────────────────────────────────────────

SCALER_FACTORIES = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


# ── Public API ───────────────────────────────────────────────────────────────


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a prototypical network (Euclidean or Poincaré).

    Pipeline:
        1. Load & preprocess data
        2. Encode labels, stratified train/val split
        3. Build ``EmbeddingNetwork`` (method-aware)
        4. Build Adam optimiser over ``model.parameters()``
        5. Training loop with LR scheduling + early stopping
        6. Post-training: compute thresholds, save checkpoint
        7. Return results dict

    Args:
        config: Full experiment configuration (loaded from YAML).

    Returns:
        Dictionary with keys: ``model``, ``prototypes``, ``thresholds``,
        ``per_class_thresholds``, ``label_encoder``, ``history``,
        ``metrics``.
    """
    console = Console()

    # ── Unpack config ────────────────────────────────────────────────
    random_state = config.get("random_state", 42)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    method = config["method"]
    data_cfg = config["data"]
    backbone_cfg = config["backbone"]
    train_cfg = config["training"]
    method_cfg = config.get(method, {})

    # Merge backbone + method-specific into a flat model config
    model_config = {**backbone_cfg, "method": method, **method_cfg}

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"\n[bold]Device:[/bold] {device}")

    # ── [1] Load data ────────────────────────────────────────────────
    console.print("\n[bold cyan][1/7] Loading data...[/bold cyan]")
    data_path = data_cfg.get("data_path")
    max_samples_per_file = data_cfg.get("max_samples_per_file", 100000)
    hold_out = data_cfg.get("hold_out")
    if not data_path:
        console.print("[red]ERROR: data.data_path is required.[/red]")
        return {"error": "No data path"}

    df = load_data(
        file_path=data_path,
        max_samples_per_file=max_samples_per_file,
        random_state=random_state,
        exclude=hold_out,
    )

    # ── [2] Configure columns & split ────────────────────────────────
    console.print("\n[bold cyan][2/7] Configuring columns...[/bold cyan]")
    column_config = configure_columns_from_dict(
        df, data_cfg.get("columns", {}),
    )

    console.print("\n[bold cyan][3/7] Extracting features & labels...[/bold cyan]")
    X, y = split_features_and_target(df, column_config)
    del df

    max_samples_per_class = data_cfg.get("max_samples_per_class")
    if max_samples_per_class and y is not None:
        console.print(f"  [bold cyan]Balancing classes (max {max_samples_per_class} per class)...[/bold cyan]")
        import pandas as pd
        indices = []
        for class_label, group_indices in y.groupby(y).groups.items():
            if len(group_indices) > max_samples_per_class:
                sampled = pd.Series(list(group_indices)).sample(n=max_samples_per_class, random_state=random_state).tolist()
                indices.extend(sampled)
            else:
                indices.extend(list(group_indices))
        
        # Shuffle indices to break any class ordering
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        X = X.loc[indices].reset_index(drop=True)
        y = y.loc[indices].reset_index(drop=True)
        console.print(f"✓ Balanced data shape: {X.shape}")

    console.print("\n[bold cyan][4/7] Preprocessing & scaling...[/bold cyan]")
    X = preprocess_data(X, handle_inf=True, handle_nan=True, fill_value=0)

    scaling_cfg = data_cfg.get("scaling", {})
    scaler_method = scaling_cfg.get("method", data_cfg.get("scaler", "standard"))

    if scaler_method == "log_quantile":
        passthrough = scaling_cfg.get("passthrough_features", [])
        feature_cols = column_config["feature_columns"]
        scale_cols = [c for c in feature_cols if c not in passthrough]
        pass_cols = [c for c in feature_cols if c in passthrough]
        
        scaler = LogQuantileScaler(
            scale_columns=scale_cols,
            passthrough_columns=pass_cols,
            n_quantiles=scaling_cfg.get("quantile_n_quantiles", 1000),
            output_distribution=scaling_cfg.get("quantile_distribution", "normal"),
            random_state=random_state,
        )
        console.print(f"  [bold cyan]Log+Quantile scaling: {len(scale_cols)} scaled, {len(pass_cols)} passthrough[/bold cyan]")
    else:
        scaler_factory = SCALER_FACTORIES.get(scaler_method, StandardScaler)
        scaler = scaler_factory()

    X_scaled = scaler.fit_transform(X)
    console.print(f"✓ Scaled data shape: {X_scaled.shape}")
    del X

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    console.print(f"✓ Classes ({num_classes}): {list(label_encoder.classes_)}")
    del y

    # Stratified train/val split
    val_split = data_cfg.get("validation_split", 0.2)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded,
            test_size=val_split, stratify=y_encoded, random_state=random_state,
        )
    except ValueError:
        console.print("[yellow]Stratified split failed — falling back to random.[/yellow]")
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_encoded,
            test_size=val_split, random_state=random_state,
        )
    del X_scaled, y_encoded

    console.print(f"  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}")

    # ── [5] Build model ──────────────────────────────────────────────
    console.print("\n[bold cyan][5/7] Building model...[/bold cyan]")
    input_dim = X_train.shape[1]
    model = EmbeddingNetwork(input_dim, num_classes, model_config).to(device)
    console.print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(
        f"  Parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    # ── [6] Training loop ────────────────────────────────────────────
    console.print("\n[bold cyan][6/7] Training...[/bold cyan]")

    epochs = train_cfg.get("epochs", 100)
    batch_size = train_cfg.get("batch_size", 2048)
    lr = train_cfg.get("learning_rate", 5e-4)
    es_patience = train_cfg.get("early_stopping_patience", 25)
    lr_patience = train_cfg.get("reduce_lr_patience", 10)
    lr_factor = train_cfg.get("reduce_lr_factor", 0.5)
    min_lr = train_cfg.get("min_learning_rate", 1e-6)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor,
        patience=lr_patience, min_lr=min_lr,
    )

    # Convert numpy → tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    del X_train, X_val, y_train, y_val

    # Print config panel
    _print_training_panel(console, method, model_config, train_cfg)

    # History tracking
    history: Dict[str, list] = {"loss": [], "val_loss": [], "learning_rate": []}

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    n_train = X_train_t.shape[0]
    steps_per_epoch = (n_train + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training", total=epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0

            # Shuffle indices
            perm = torch.randperm(n_train, device=device)

            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min(start + batch_size, n_train)
                idx = perm[start:end]

                x_batch = X_train_t[idx]
                y_batch = y_train_t[idx]

                optimizer.zero_grad(set_to_none=True)

                embeddings, recons = model(x_batch, return_recon=True)
                loss = _compute_loss(
                    embeddings, y_batch, model.prototypes,
                    method, model_config,
                )
                
                # Add Reconstruction Loss
                recon_weight = model_config.get("reconstruction_weight", 0.0)
                if recon_weight > 0.0 and recons is not None:
                    mse_loss = torch.nn.functional.mse_loss(recons, x_batch)
                    loss = loss + recon_weight * mse_loss
                    
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()


                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / steps_per_epoch

            # ── Validation ───────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                val_emb, val_recons = model(X_val_t, return_recon=True)
                val_loss_tensor = _compute_loss(
                    val_emb, y_val_t, model.prototypes,
                    method, model_config,
                )
                
                recon_weight = model_config.get("reconstruction_weight", 0.0)
                if recon_weight > 0.0 and val_recons is not None:
                    val_mse = torch.nn.functional.mse_loss(val_recons, X_val_t)
                    val_loss_tensor = val_loss_tensor + recon_weight * val_mse
                    
                val_loss = val_loss_tensor.item()

            current_lr = optimizer.param_groups[0]["lr"]
            history["loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(current_lr)

            scheduler.step(val_loss)

            # ── Early stopping ───────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            progress.update(
                task, advance=1,
                description=(
                    f"Epoch {epoch}/{epochs} | "
                    f"loss={avg_train_loss:.4f} val={val_loss:.4f} "
                    f"lr={current_lr:.2e} "
                    f"[best={best_val_loss:.4f}@{best_epoch}]"
                ),
            )

            if patience_counter >= es_patience:
                console.print(
                    f"\n[yellow]Early stopping at epoch {epoch} "
                    f"(best={best_val_loss:.4f} @ epoch {best_epoch})[/yellow]"
                )
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    console.print(
        f"\n✓ Restored best weights from epoch {best_epoch} "
        f"(val_loss={best_val_loss:.4f})"
    )

    # ── [7] Post-training: thresholds & checkpoint ───────────────────
    console.print("\n[bold cyan][7/7] Computing thresholds & saving...[/bold cyan]")

    model.eval()
    with torch.no_grad():
        all_emb_list = []
        chunk_size = 4096
        for i in range(0, X_train_t.shape[0], chunk_size):
            all_emb_list.append(model(X_train_t[i:i+chunk_size]))
        all_emb = torch.cat(all_emb_list, dim=0)

    if method == "poincare":
        inference_prototypes = model.prototypes.detach()
        console.print(f"  ✓ Using dynamically anchored prototypes for {num_classes} classes")
    else:
        inference_prototypes = model.prototypes.detach()

    thresholds, per_class_thresholds = _compute_thresholds(
        all_emb, y_train_t, inference_prototypes,
        method, model_config, num_classes,
        percentile=train_cfg.get("threshold_percentile", 95.0),
    )

    # Print thresholds
    console.print(f"\n  Global threshold τ: {thresholds['global']:.4f}")
    if "origin" in thresholds:
        console.print(f"  Origin threshold τ₀: {thresholds['origin']:.4f}")
    for cls_idx, tau_k in sorted(per_class_thresholds.items()):
        cls_name = label_encoder.classes_[cls_idx]
        console.print(f"    τ({cls_name}) = {tau_k:.4f}")

    # Build serialisable metrics
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history["loss"][-1]),
        "num_classes": num_classes,
        "embedding_dim": model_config["embedding_dim"],
        "method": method,
        "global_threshold": float(thresholds["global"]),
    }
    if "origin" in thresholds:
        metrics["origin_threshold"] = float(thresholds["origin"])

    # Save checkpoint
    project_root = Path(__file__).parent.parent
    model_dir = project_root / config.get("model_dir", "models")
    results_dir = project_root / config.get("results_dir", "results") / method
    results_dir.mkdir(parents=True, exist_ok=True)

    # Serialisable model config (strip non-JSON types)
    serialisable_config = {k: v for k, v in model_config.items()}
    serialisable_config["max_samples_per_file"] = max_samples_per_file
    serialisable_config["hold_out"] = data_cfg.get("hold_out")

    saved_paths = save_checkpoint(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        column_config=column_config,
        model_config=serialisable_config,
        metrics=metrics,
        save_dir=model_dir,
        method_name=method,
        additional_artifacts={
            "thresholds.pkl": thresholds,
            "per_class_thresholds.pkl": per_class_thresholds,
            "inference_prototypes.pkl": inference_prototypes.cpu(),
        },
    )

    # ── Save training history plot ───────────────────────────────────
    plots_cfg = config.get("plots", {})
    if plots_cfg.get("save_training_plots", False):
        _save_training_plot(history, results_dir)

    # Cleanup
    del X_train_t, y_train_t, X_val_t, y_val_t, all_emb
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.print("\n" + "=" * 80)
    console.print("[bold green]TRAINING COMPLETED![/bold green]")
    console.print("=" * 80)
    console.print(f"  Method: {method}")
    console.print(f"  Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
    console.print(f"  Model saved: {saved_paths['version_dir']}")

    return {
        "model": model,
        "prototypes": model.prototypes,
        "inference_prototypes": inference_prototypes,
        "thresholds": thresholds,
        "per_class_thresholds": per_class_thresholds,
        "label_encoder": label_encoder,
        "history": history,
        "metrics": metrics,
        "saved_paths": saved_paths,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _compute_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    method: str,
    config: Dict[str, Any],
) -> torch.Tensor:
    """Dispatch to the correct loss function based on method."""
    if method == "euclidean":
        return euclidean_prototypical_loss(
            embeddings, labels, prototypes,
            distance_metric=config.get("distance_metric", "squared_euclidean"),
            temperature=config.get("distance_temperature", 0.1),
            center_loss_radius=config.get("center_loss_radius", 0.0),
            center_loss_weight=config.get("center_loss_weight", 0.1),
            supcon_weight=config.get("supcon_weight", 0.0),
            supcon_temperature=config.get("supcon_temperature", 0.07),
        )
    elif method == "poincare":
        return hyperbolic_prototypical_loss(
            embeddings, labels, prototypes,
            curvature=config.get("curvature", 1.0),
            push_margin=config.get("push_margin", 4.0),
            push_weight=config.get("push_weight", 1.0),
            origin_pull_weight=config.get("origin_pull_weight", 0.0),
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_thresholds(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    method: str,
    config: Dict[str, Any],
    num_classes: int,
    percentile: float = 95.0,
) -> tuple:
    """
    Compute global and per-class distance thresholds.

    All distance computation is GPU-resident.  Only the final
    ``np.percentile`` call touches CPU.

    Returns:
        (thresholds_dict, per_class_dict)
    """
    if method == "euclidean":
        diff = embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=2)
        metric = config.get("distance_metric", "squared_euclidean")
        if metric == "euclidean":
            all_dists = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
        else:
            all_dists = dist_sq
    else:
        all_dists = poincare_distance(
            embeddings.unsqueeze(1),
            prototypes.unsqueeze(0),
            c=config.get("curvature", 1.0),
        )

    # Find nearest prototype for all training samples
    min_dists, nearest_idx = all_dists.min(dim=1)
    min_dists_np = min_dists.cpu().numpy()
    nearest_idx_np = nearest_idx.cpu().numpy()

    # Global threshold (based on ALL minimum distances)
    global_threshold = float(np.percentile(min_dists_np, percentile))
    thresholds = {"global": global_threshold}

    # Poincaré: also compute origin-distance threshold
    if method == "poincare":
        c = config.get("curvature", 1.0)
        origin_dists = origin_distance(embeddings, c=c).cpu().numpy()
        thresholds["origin"] = float(np.percentile(origin_dists, 100.0 - percentile))

    # Per-class thresholds (based on samples assigned to each cluster)
    per_class: Dict[int, float] = {}
    for cls_idx in range(num_classes):
        mask = nearest_idx_np == cls_idx
        if mask.any():
            per_class[cls_idx] = float(
                np.percentile(min_dists_np[mask], percentile)
            )
        else:
            per_class[cls_idx] = 0.0

    return thresholds, per_class


def _print_training_panel(
    console: Console,
    method: str,
    model_config: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> None:
    """Print a Rich panel summarising the training configuration."""
    lines = [f"[bold]Method:[/bold] {method.upper()}"]

    if method == "euclidean":
        lines.append(
            f"[bold]Distance:[/bold] {model_config.get('distance_metric', 'sq_euclidean')} "
            f"(τ={model_config.get('distance_temperature', 0.1)})"
        )
        r = model_config.get("center_loss_radius", 0.0)
        if r > 0:
            lines.append(
                f"[bold]Center Loss:[/bold] R={r}, "
                f"λ={model_config.get('center_loss_weight', 0.1)}"
            )
        else:
            lines.append("[bold]Center Loss:[/bold] disabled")

        sc = model_config.get("supcon_weight", 0.0)
        if sc > 0:
            lines.append(
                f"[bold]SupCon:[/bold] λ={sc}, "
                f"τ_sc={model_config.get('supcon_temperature', 0.07)}"
            )
        else:
            lines.append("[bold]SupCon:[/bold] disabled")

    elif method == "poincare":
        lines.append(f"[bold]Curvature:[/bold] c={model_config.get('curvature', 1.0)}")
        lines.append(
            f"[bold]Prototype radius:[/bold] "
            f"{model_config.get('prototype_placement_radius', 0.8)}"
        )
        lines.append("[bold]Loss:[/bold] direct distance mean")

    lines.append(
        f"[bold]Optimizer:[/bold] Adam (lr={train_cfg.get('learning_rate', 5e-4)}, "
        f"clip=1.0)"
    )
    lines.append(
        f"[bold]Schedule:[/bold] ReduceLROnPlateau "
        f"(patience={train_cfg.get('reduce_lr_patience', 10)})"
    )
    lines.append(
        f"[bold]Early stop:[/bold] patience={train_cfg.get('early_stopping_patience', 25)}"
    )

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Training Configuration[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))


def _save_training_plot(
    history: Dict[str, list],
    results_dir: Path,
) -> None:
    """Save training loss curve as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(history["loss"], label="Train Loss", linewidth=1.5)
        ax.plot(history["val_loss"], label="Val Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training History")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = results_dir / "training_history.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved training plot → {path}")
    except Exception as e:
        print(f"  [WARNING] Could not save training plot: {e}")
