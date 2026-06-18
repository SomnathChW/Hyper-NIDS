"""
Data loading and preprocessing utilities.

Handles CSV/Parquet loading (single file or folder), column role
configuration, feature/target splitting, and inf/NaN cleaning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


# ── Column configuration ────────────────────────────────────────────────────


def configure_columns_from_dict(
    df: pd.DataFrame,
    column_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a structured column configuration from a role dictionary.

    Args:
        df: Input dataframe whose columns are inspected.
        column_config: Mapping of ``{column_name: {role: "feature"|"target"|"drop"}}``.
            Columns absent from the mapping default to *feature* if numeric,
            or *drop* otherwise.

    Returns:
        Dictionary with keys ``feature_columns``, ``target_column``,
        ``dropped_columns``, ``numerical_columns``.

    Raises:
        ValueError: If more than one target column is specified.
    """
    numerical_cols = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"],
    ).columns.tolist()

    config: Dict[str, Any] = {
        "feature_columns": [],
        "target_column": None,
        "dropped_columns": [],
        "numerical_columns": [],
    }

    target_set = False

    for col in df.columns:
        if col not in column_config:
            # Default: numeric → feature, categorical → drop
            if col in numerical_cols:
                config["feature_columns"].append(col)
                config["numerical_columns"].append(col)
            else:
                config["dropped_columns"].append(col)
            continue

        role = column_config[col].get("role", "feature")

        if role == "target":
            if target_set:
                raise ValueError(
                    f"Multiple target columns specified: "
                    f"{config['target_column']} and {col}"
                )
            config["target_column"] = col
            target_set = True
        elif role == "drop":
            config["dropped_columns"].append(col)
        elif role == "feature":
            config["feature_columns"].append(col)
            if col in numerical_cols:
                config["numerical_columns"].append(col)

    print("\n" + "=" * 80)
    print("COLUMN CONFIGURATION")
    print("=" * 80)
    print(f"✓ Feature columns: {len(config['feature_columns'])}")
    print(f"✓ Target column: {config['target_column']}")
    print(f"✓ Dropped columns: {len(config['dropped_columns'])}")
    print("=" * 80)

    return config


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data(
    file_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load data from a file, a folder of files, or generate demo data.

    Args:
        file_path: Path to a CSV/Parquet file **or** a directory containing
            such files.  If ``None``, synthetic demo data is generated.
        sample_size: Number of samples when generating demo data.
        random_state: Random seed for demo data.

    Returns:
        Loaded (or generated) DataFrame.
    """
    if file_path is not None:
        file_path = Path(file_path)

        if file_path.is_dir():
            print(f"[INFO] Loading from folder: {file_path}")
            csv_files = list(file_path.glob("*.csv"))
            parquet_files = list(file_path.glob("*.parquet"))
            all_files = csv_files + parquet_files

            if not all_files:
                raise ValueError(
                    f"No CSV or parquet files found in directory: {file_path}"
                )

            print(
                f"  Found {len(csv_files)} CSV file(s) "
                f"and {len(parquet_files)} parquet file(s)"
            )

            dfs = []
            for i, data_file in enumerate(all_files, 1):
                print(
                    f"  [{i}/{len(all_files)}] Loading: {data_file.name}...",
                    end=" ",
                )
                if data_file.suffix == ".csv":
                    temp_df = pd.read_csv(data_file)
                elif data_file.suffix == ".parquet":
                    temp_df = pd.read_parquet(data_file)
                else:
                    continue  # skip unsupported
                print(f"✓ ({temp_df.shape[0]} rows)")
                dfs.append(temp_df)

            df = pd.concat(dfs, ignore_index=True)
            print(
                f"\n✓ Combined {len(all_files)} files into "
                f"{df.shape[0]} rows, {df.shape[1]} columns"
            )

        elif file_path.is_file():
            print(f"[INFO] Loading from file: {file_path}")
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}"
                )
            print(f"✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            raise FileNotFoundError(f"Path does not exist: {file_path}")

        return df

    # ── Generate demo data ───────────────────────────────────────────────
    n_samples = sample_size or 300
    n_outliers = int(0.15 * n_samples)
    n_inliers = n_samples - n_outliers

    rng = np.random.RandomState(random_state)

    covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
    cluster_1 = 0.4 * rng.randn(n_inliers // 2, 2) @ covariance + np.array(
        [2, 2]
    )
    cluster_2 = 0.3 * rng.randn(n_inliers // 2, 2) + np.array([-2, -2])
    outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

    X = np.concatenate([cluster_1, cluster_2, outliers])
    y = np.concatenate(
        [np.ones(n_inliers, dtype=int), -np.ones(n_outliers, dtype=int)]
    )

    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    df["label"] = y
    print(
        f"✓ Generated demo data: {df.shape[0]} rows, {df.shape[1]} columns"
    )
    return df


# ── Feature / target splitting ───────────────────────────────────────────────


def split_features_and_target(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Split a DataFrame into features (X) and target (y).

    Args:
        df: Input dataframe.
        config: Column configuration produced by
            :func:`configure_columns_from_dict`.

    Returns:
        ``(X, y)`` where *y* is ``None`` if no target column is present.
    """
    if not isinstance(config, dict):
        raise ValueError("Column configuration must be a dictionary.")
    if "feature_columns" not in config:
        raise ValueError(
            "Column configuration missing required key: 'feature_columns'."
        )
    if "target_column" not in config:
        raise ValueError(
            "Column configuration missing required key: 'target_column'."
        )

    feature_columns = config["feature_columns"]
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("'feature_columns' must be a non-empty list.")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    X = df[feature_columns].copy()

    target_column = config["target_column"]
    y = (
        df[target_column].copy()
        if target_column and target_column in df.columns
        else None
    )

    print(f"\n✓ Data split applied:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape if y is not None else 'None'}")

    return X, y


# ── Preprocessing ────────────────────────────────────────────────────────────


def preprocess_data(
    X: pd.DataFrame,
    handle_inf: bool = True,
    handle_nan: bool = True,
    fill_value: float = 0,
) -> pd.DataFrame:
    """
    Clean a feature DataFrame by replacing infinities and filling NaNs.

    Args:
        X: Input features.
        handle_inf: Replace ``±inf`` with ``NaN``.
        handle_nan: Fill ``NaN`` values with *fill_value*.
        fill_value: Replacement value for NaNs.

    Returns:
        Cleaned DataFrame (copy of the input).
    """
    X_clean = X.copy()

    if handle_inf:
        inf_count = X.isin([np.inf, -np.inf]).sum().sum()
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        if inf_count > 0:
            print(
                f"  [WARNING] Replaced {inf_count} infinite values with NaN"
            )

    if handle_nan:
        nan_count = X_clean.isnull().sum().sum()
        if nan_count > 0:
            X_clean = X_clean.fillna(fill_value)
            print(
                f"  [WARNING] Filled {nan_count} NaN values with {fill_value}"
            )

    return X_clean
