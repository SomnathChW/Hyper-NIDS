import sys
import gc
from pathlib import Path
from typing import Dict, Any
import pickle

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from rich.console import Console

from modules.model import EmbeddingNetwork
from modules.utils.data_utils import (
    configure_columns_from_dict,
    load_data,
    preprocess_data,
    split_features_and_target,
)
from modules.utils.model_utils import load_checkpoint
from modules.train import _compute_thresholds

console = Console()

def run_recalculate(method: str, percentile: float):
    """
    Recalculates Open-Set detection thresholds for a saved model checkpoint
    using the original training data and a new percentile.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"\n[bold]Device:[/bold] {device}")
    
    # ── [1] Load Checkpoint ─────────────────────────────────────────
    console.print("\n[bold cyan][1/4] Loading latest checkpoint...[/bold cyan]")
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "models"
    
    artifacts = load_checkpoint(model_dir, method_name=method)
    
    if "experiment_config" in artifacts:
        console.print("  ✓ Found experiment config embedded in checkpoint")
        config = artifacts["experiment_config"]
    else:
        console.print("[red]ERROR: No experiment_config.pkl found in checkpoint. Cannot recalculate thresholds for older models.[/red]")
        sys.exit(1)

    random_state = config.get("random_state", 42)
    data_cfg = config.get("data", {})
    
    # ── [2] Load Data (same as training) ────────────────────────────
    console.print(f"\n[bold cyan][2/4] Loading training data...[/bold cyan]")
    data_path = data_cfg.get("data_path")
    max_samples_per_file = data_cfg.get("max_samples_per_file", 100000)
    hold_out = data_cfg.get("hold_out")
    
    if not data_path:
        console.print("[red]ERROR: data.data_path is required in config.[/red]")
        sys.exit(1)

    df = load_data(
        file_path=data_path,
        max_samples_per_file=max_samples_per_file,
        random_state=random_state,
        exclude=hold_out,
    )

    column_config = configure_columns_from_dict(df, data_cfg.get("columns", {}))
    X, y = split_features_and_target(df, column_config)
    del df

    max_samples_per_class = data_cfg.get("max_samples_per_class")
    if max_samples_per_class and y is not None:
        import pandas as pd
        indices = []
        for class_label, group_indices in y.groupby(y).groups.items():
            if len(group_indices) > max_samples_per_class:
                sampled = pd.Series(list(group_indices)).sample(n=max_samples_per_class, random_state=random_state).tolist()
                indices.extend(sampled)
            else:
                indices.extend(list(group_indices))
        
        np.random.seed(random_state)
        np.random.shuffle(indices)
        X = X.loc[indices].reset_index(drop=True)
        y = y.loc[indices].reset_index(drop=True)

    X = preprocess_data(X, handle_inf=True, handle_nan=True, fill_value=0)

    # Setup for model evaluation
    state_dict = artifacts["state_dict"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    model_config = artifacts["model_config"]
    version_dir = Path(artifacts["version_dir"])
    
    X_scaled = scaler.transform(X)
    del X
    y_encoded = label_encoder.transform(y)
    num_classes = len(label_encoder.classes_)

    val_split = data_cfg.get("validation_split", 0.2)
    X_train, _, y_train, _ = train_test_split(
        X_scaled, y_encoded, test_size=val_split,
        random_state=random_state, stratify=y_encoded,
    )
    del X_scaled, y_encoded
    
    first_weight_key = [k for k in state_dict if "backbone.0.weight" in k][0]
    input_dim = state_dict[first_weight_key].shape[1]
    
    model = EmbeddingNetwork(input_dim, num_classes, model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── [3] Generate Embeddings ─────────────────────────────────────
    console.print("\n[bold cyan][3/4] Generating embeddings...[/bold cyan]")
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    del X_train
    
    with torch.no_grad():
        all_emb_list = []
        chunk_size = 4096
        for i in range(0, X_train_t.shape[0], chunk_size):
            all_emb_list.append(model(X_train_t[i:i+chunk_size]))
        all_emb = torch.cat(all_emb_list, dim=0)

    inference_prototypes = artifacts.get("inference_prototypes", model.prototypes)
    if inference_prototypes is not None and hasattr(inference_prototypes, "to"):
        inference_prototypes = inference_prototypes.to(device)

    # ── [4] Recalculate Thresholds ──────────────────────────────────
    console.print("\n[bold cyan][4/4] Computing new thresholds...[/bold cyan]")
    thresholds, per_class_thresholds = _compute_thresholds(
        all_emb, y_train_t, inference_prototypes,
        method, model_config, num_classes,
        percentile=percentile,
    )

    console.print(f"\n  Global threshold τ: {thresholds['global']:.4f}")
    if "origin" in thresholds:
        console.print(f"  Origin threshold τ₀: {thresholds['origin']:.4f}")
    for cls_idx, tau_k in sorted(per_class_thresholds.items()):
        cls_name = label_encoder.classes_[cls_idx]
        console.print(f"    τ({cls_name}) = {tau_k:.4f}")

    # Overwrite the pickled thresholds
    thresh_path = version_dir / "thresholds.pkl"
    with open(thresh_path, "wb") as f:
        pickle.dump(thresholds, f)
        
    per_thresh_path = version_dir / "per_class_thresholds.pkl"
    with open(per_thresh_path, "wb") as f:
        pickle.dump(per_class_thresholds, f)
        
    console.print(f"\n[bold green]✓ Successfully updated thresholds for {method} at {percentile}th percentile.[/bold green]")
