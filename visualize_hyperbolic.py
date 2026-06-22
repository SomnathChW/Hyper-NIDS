import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from modules.poincare_math import log_map_zero, exp_map_zero, poincare_distance
from modules.utils.model_utils import load_checkpoint
from modules.utils.data_utils import load_data, configure_columns_from_dict, split_features_and_target, preprocess_data

def hyperbolic_pca(embeddings_tensor, c=1.0, n_components=2):
    """
    Performs Hyperbolic PCA by mapping to the origin's tangent space,
    doing Euclidean PCA, and mapping back.
    """
    # 1. Map to tangent space at origin (Euclidean space)
    tangent_vectors = log_map_zero(embeddings_tensor, c=c)
    tangent_np = tangent_vectors.cpu().numpy()
    
    # 2. Perform Euclidean PCA
    pca = PCA(n_components=n_components)
    tangent_2d = pca.fit_transform(tangent_np)
    tangent_2d_tensor = torch.tensor(tangent_2d, dtype=torch.float32, device=embeddings_tensor.device)
    
    # 3. Map back to 2D Poincaré disk
    embeddings_2d = exp_map_zero(tangent_2d_tensor, c=c)
    
    return embeddings_2d.cpu().numpy(), pca

def main():
    parser = argparse.ArgumentParser(description="Visualize Hyperbolic NIDS in 2D using H-PCA")
    parser.add_argument("--method", default="poincare", help="Method to load")
    parser.add_argument("--split", default="val", help="Data split to visualize (train/val)")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    model_dir = project_root / "models"
    
    print("Loading checkpoint...")
    try:
        artifacts = load_checkpoint(model_dir, method_name=args.method)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    model_config = artifacts["model_config"]
    if model_config.get("method", args.method) != "poincare":
        print("This visualization is specific to the Poincaré method.")
        sys.exit(1)
        
    c = model_config.get("curvature", 1.0)
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    column_config = artifacts["column_config"]
    experiment_config = artifacts.get("experiment_config", {})
    data_cfg = experiment_config.get("data", {})
    
    # Load data
    data_path = data_cfg.get("data_path", "/kaggle/working/dataset")
    print(f"Loading data from {data_path}...")
    df = load_data(
        file_path=data_path,
        max_samples_per_file=20000, # Load a smaller subset for visualization
        exclude=data_cfg.get("hold_out")
    )
    
    X, y = split_features_and_target(df, column_config)
    X = preprocess_data(X, handle_inf=True, handle_nan=True, fill_value=0)
    X_scaled = scaler.transform(X)
    y_encoded = label_encoder.transform(y)
    
    # Rebuild Model
    from modules.model import EmbeddingNetwork
    first_weight_key = [k for k in artifacts["state_dict"] if "backbone.0.weight" in k][0]
    input_dim = artifacts["state_dict"][first_weight_key].shape[1]
    num_classes = len(label_encoder.classes_)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingNetwork(input_dim, num_classes, model_config).to(device)
    model.load_state_dict(artifacts["state_dict"])
    model.eval()
    
    print("Generating embeddings...")
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        embeddings = model(X_t)
        
    fixed_prototypes = model.prototypes
    inference_prototypes = artifacts.get("inference_prototypes", fixed_prototypes).to(device)
    
    print("Performing Hyperbolic PCA...")
    # Include prototypes in PCA to project them to the same 2D space
    num_embs = embeddings.shape[0]
    all_points = torch.cat([embeddings, fixed_prototypes, inference_prototypes], dim=0)
    all_points_2d, _ = hyperbolic_pca(all_points, c=c)
    
    emb_2d = all_points_2d[:num_embs]
    fixed_2d = all_points_2d[num_embs:num_embs+num_classes]
    inf_2d = all_points_2d[num_embs+num_classes:]
    
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw Poincaré disk boundary
    boundary = plt.Circle((0, 0), 1.0 / np.sqrt(c), color='black', fill=False, linewidth=2)
    ax.add_patch(boundary)
    
    # Draw fixed prototype placement radius
    p_radius = model_config.get("prototype_placement_radius", 0.5)
    inner_circle = plt.Circle((0, 0), p_radius, color='gray', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(inner_circle)
    
    # Plot embeddings
    sns.scatterplot(
        x=emb_2d[:, 0], y=emb_2d[:, 1],
        hue=label_encoder.inverse_transform(y_encoded),
        palette='tab10', alpha=0.2, s=15, ax=ax, edgecolor='none'
    )
    
    # Plot Fixed Prototypes (Stars)
    ax.scatter(fixed_2d[:, 0], fixed_2d[:, 1], marker='*', s=300, c='black', label='Fixed Prototypes (r=0.5)', zorder=5)
    
    # Plot Inference Prototypes (Fréchet Means) (X's)
    ax.scatter(inf_2d[:, 0], inf_2d[:, 1], marker='X', s=200, c='red', label='Inference Prototypes (Fréchet Mean)', zorder=5)
    
    # Draw arrows from Fixed to Inference
    for i in range(num_classes):
        ax.annotate(
            "", xy=(inf_2d[i, 0], inf_2d[i, 1]), xycoords='data',
            xytext=(fixed_2d[i, 0], fixed_2d[i, 1]), textcoords='data',
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.7)
        )
        
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title("Hyperbolic PCA Visualization (True Poincaré Geometry)\nNotice how Fréchet Means (X) drift towards the boundary compared to Fixed Prototypes (*)")
    ax.legend(loc='upper right')
    ax.axis('off')
    
    out_path = project_root / "hyperbolic_pca_visualization.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
