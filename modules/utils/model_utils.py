"""
Model checkpoint save / load utilities.

Uses PyTorch's native ``torch.save`` / ``torch.load`` for model state and
pickle for sklearn artifacts.  Follows TriAD's ``latest.json`` pointer
pattern for version management.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# ── Validation ───────────────────────────────────────────────────────────────


def validate_model_artifacts(load_dir: str | Path, method_name: str, version: str = "latest"):
    """
    Validate that all expected model files exist before attempting to load.
    """
    load_dir = Path(load_dir)
    method_dir = load_dir / method_name
    
    # Resolve version
    if version == 'latest':
        latest_pointer = method_dir / 'latest.json'
        if not latest_pointer.exists():
            raise FileNotFoundError(f"Latest pointer file not found: {latest_pointer}")
        with open(latest_pointer, 'r') as f:
            version_name = json.load(f)['version']
        version_dir = method_dir / version_name
    else:
        version_dir = method_dir / version
    
    if not version_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")
    
    expected_files = [
        'model.pt',
        'scaler.pkl',
        'label_encoder.pkl',
        'column_config.json',
        'model_config.json',
        'metrics.json',
    ]
    
    missing_files = []
    found_files = []
    errors = []
    
    for filename in expected_files:
        filepath = version_dir / filename
        if filepath.exists():
            found_files.append(filename)
        else:
            missing_files.append(filename)
            errors.append(f"Missing required file: {filename}")
    
    is_valid = len(missing_files) == 0
    
    validation_results = {
        'version_dir': version_dir,
        'missing_files': missing_files,
        'found_files': found_files,
        'errors': errors
    }
    
    return is_valid, validation_results


def print_validation_report(validation_results: Dict[str, Any], method_name: str):
    """
    Print a formatted validation report.
    """
    print(f"\n{'='*70}")
    print(f"Model Artifact Validation Report - {method_name.upper()}")
    print(f"{'='*70}")
    print(f"Version Directory: {validation_results['version_dir']}")
    print(f"\n✓ Found Files ({len(validation_results['found_files'])}):")
    for f in validation_results['found_files']:
        print(f"  ✓ {f}")
    
    if validation_results['missing_files']:
        print(f"\n✗ Missing Files ({len(validation_results['missing_files'])}):")
        for f in validation_results['missing_files']:
            print(f"  ✗ {f}")
        print(f"\n{'='*70}")
        print("VALIDATION FAILED - Cannot load model")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("VALIDATION PASSED - All required files present")
        print(f"{'='*70}\n")


# ── Save ─────────────────────────────────────────────────────────────────────


def save_checkpoint(
    model: torch.nn.Module,
    scaler: Any,
    label_encoder: Any,
    column_config: Dict[str, Any],
    model_config: Dict[str, Any],
    metrics: Dict[str, Any],
    save_dir: str | Path,
    method_name: str,
    additional_artifacts: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Save all training artifacts for a single experiment run.

    Creates a timestamped version directory under
    ``<save_dir>/<method_name>/`` and writes:

    * ``model.pt``           – full ``state_dict`` (backbone + prototypes)
    * ``scaler.pkl``         – fitted sklearn scaler
    * ``label_encoder.pkl``  – fitted LabelEncoder
    * ``column_config.json`` – column roles
    * ``model_config.json``  – architecture / loss hyperparameters
    * ``metrics.json``       – final training metrics
    * any extra artifacts from *additional_artifacts*

    A ``latest.json`` pointer is updated in the method directory so that
    :func:`load_checkpoint` can find the most recent version.

    Args:
        model: Trained ``EmbeddingNetwork``.
        scaler: Fitted feature scaler (StandardScaler, etc.).
        label_encoder: Fitted ``LabelEncoder``.
        column_config: Column role configuration dict.
        model_config: Model / loss hyperparameters dict.
        metrics: Training metrics dict.
        save_dir: Root model directory (e.g. ``models/``).
        method_name: ``"euclidean"`` or ``"poincare"`` – used as sub-folder.
        additional_artifacts: Optional ``{filename: object}`` pairs to pickle.

    Returns:
        Dict mapping artifact names to their absolute paths.
    """
    method_dir = Path(save_dir) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = method_dir / timestamp
    version_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, str] = {"version_dir": str(version_dir)}

    # Model state dict (includes backbone weights + prototype nn.Parameter / buffer)
    model_path = version_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    saved_paths["model"] = str(model_path)
    print(f"  ✓ Saved model state dict: {model_path.name}")

    # Scaler
    scaler_path = version_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    saved_paths["scaler"] = str(scaler_path)
    print(f"  ✓ Saved scaler: {scaler_path.name}")

    # Label encoder
    le_path = version_dir / "label_encoder.pkl"
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)
    saved_paths["label_encoder"] = str(le_path)
    print(f"  ✓ Saved label encoder: {le_path.name}")

    # Column config
    cc_path = version_dir / "column_config.json"
    with open(cc_path, "w") as f:
        json.dump(column_config, f, indent=2)
    saved_paths["column_config"] = str(cc_path)

    # Model config
    mc_path = version_dir / "model_config.json"
    with open(mc_path, "w") as f:
        json.dump(model_config, f, indent=2)
    saved_paths["model_config"] = str(mc_path)

    # Metrics
    metrics_path = version_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    saved_paths["metrics"] = str(metrics_path)

    # Additional artifacts (thresholds, per-class thresholds, etc.)
    if additional_artifacts:
        for filename, artifact in additional_artifacts.items():
            artifact_path = version_dir / filename
            with open(artifact_path, "wb") as f:
                pickle.dump(artifact, f)
            saved_paths[filename] = str(artifact_path)
            print(f"  ✓ Saved {filename}")

    # Update latest pointer
    latest_pointer = method_dir / "latest.json"
    with open(latest_pointer, "w") as f:
        json.dump({"version": version_dir.name}, f, indent=2)

    print(f"\n✓ Artifacts saved to: {version_dir}")
    print(f"✓ Updated latest pointer: {latest_pointer}")

    return saved_paths


# ── Load ─────────────────────────────────────────────────────────────────────


def load_checkpoint(
    load_dir: str | Path,
    method_name: str,
    version: str = "latest",
) -> Dict[str, Any]:
    """
    Load a previously saved checkpoint.

    Args:
        load_dir: Root model directory (e.g. ``models/``).
        method_name: ``"euclidean"`` or ``"poincare"``.
        version: ``"latest"`` or a specific timestamp string.

    Returns:
        Dictionary with keys:
        ``state_dict``, ``scaler``, ``label_encoder``,
        ``column_config``, ``model_config``, ``metrics``,
        and any additional pickled artifacts found.

    Raises:
        FileNotFoundError: If the version directory or required files
            are missing.
    """
    # Validate artifacts before loading
    is_valid, validation_results = validate_model_artifacts(load_dir, method_name, version)
    
    if not is_valid:
        print_validation_report(validation_results, method_name)
        raise FileNotFoundError(
            f"Missing required model files. Found {len(validation_results['found_files'])}/{len(validation_results['found_files']) + len(validation_results['missing_files'])} files. "
            f"Missing: {', '.join(validation_results['missing_files'])}"
        )
    
    version_dir = validation_results['version_dir']
    print(f"Loading checkpoint from: {version_dir}")
    artifacts: Dict[str, Any] = {"version_dir": str(version_dir)}

    # Model state dict
    model_path = version_dir / "model.pt"
    if model_path.exists():
        artifacts["state_dict"] = torch.load(
            model_path, map_location="cpu", weights_only=True,
        )
        print("  ✓ Loaded model state dict")

    # Scaler
    scaler_path = version_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            artifacts["scaler"] = pickle.load(f)
        print("  ✓ Loaded scaler")

    # Label encoder
    le_path = version_dir / "label_encoder.pkl"
    if le_path.exists():
        with open(le_path, "rb") as f:
            artifacts["label_encoder"] = pickle.load(f)
        print("  ✓ Loaded label encoder")

    # Column config
    cc_path = version_dir / "column_config.json"
    if cc_path.exists():
        with open(cc_path, "r") as f:
            artifacts["column_config"] = json.load(f)
        print("  ✓ Loaded column config")

    # Model config
    mc_path = version_dir / "model_config.json"
    if mc_path.exists():
        with open(mc_path, "r") as f:
            artifacts["model_config"] = json.load(f)
        print("  ✓ Loaded model config")

    # Metrics
    metrics_path = version_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            artifacts["metrics"] = json.load(f)
        print("  ✓ Loaded metrics")

    # Load any additional pickled artifacts
    for pkl_file in version_dir.glob("*.pkl"):
        basename = pkl_file.stem
        if basename not in artifacts:
            with open(pkl_file, "rb") as f:
                artifacts[basename] = pickle.load(f)
            print(f"  ✓ Loaded {pkl_file.name}")

    return artifacts
