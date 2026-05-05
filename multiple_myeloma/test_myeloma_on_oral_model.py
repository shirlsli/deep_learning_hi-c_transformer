"""
Run inference on preprocessed myeloma data using the original oral cancer checkpoint.
No fine-tuning, just zero-shot evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

ORAL_INPUT_DIM = 76
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1


class HiCTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


def project_with_oral_pca(data_tensor, pca_model_path):
    """Project data to oral PCA space (76D)."""
    pca_obj = torch.load(pca_model_path, map_location="cpu", weights_only=False)
    components = torch.tensor(np.asarray(pca_obj["components"], dtype=np.float32))
    mean = torch.tensor(np.asarray(pca_obj["mean"], dtype=np.float32))

    if data_tensor.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Feature dim {data_tensor.shape[1]} does not match oral PCA input dim {mean.shape[0]}"
        )

    centered = data_tensor - mean
    projected = centered @ components.T
    return projected.float()


def main():
    # Paths
    oral_checkpoint_path = "/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/checkpoints_20260329_133858/best_model.pt"
    oral_pca_path = "/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_133858/pca_model.pt"
    myeloma_test_data_path = "/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_data.pt"  # Raw 77405D
    myeloma_test_labels_path = "/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_labels.pt"
    myeloma_test_names_path = "/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_names.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load oral checkpoint
    print("Loading oral cancer checkpoint (epoch 99)...")
    checkpoint = torch.load(oral_checkpoint_path, map_location=device)
    
    # Extract metadata if present
    epoch = checkpoint.get("epoch", "?")
    val_pearson = checkpoint.get("val_pearson", "?")
    # Handle both nested (model_state) and flat state_dict formats
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    
    print(f"  Checkpoint metadata: epoch={epoch}, val_pearson={val_pearson}")

    # Initialize model and load weights
    # Note: oral checkpoint has regression head (output_dim=1)
    model = HiCTransformer(
        input_dim=ORAL_INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=1,  # Regression (oral model was trained on regression)
    ).to(device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Loaded successfully (regression head with 1 output)\n")

    # Load myeloma test data
    print("Loading myeloma test data...")
    X_test = torch.load(myeloma_test_data_path, map_location="cpu").float()
    y_test = torch.load(myeloma_test_labels_path, map_location="cpu").long()
    
    with open(myeloma_test_names_path) as f:
        sample_names = [line.strip() for line in f]
    
    print(f"✓ Loaded {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  Sample names: {sample_names}")
    print(f"  Labels: {y_test.tolist()}\n")

    # Project if needed (in case raw 77405D data was loaded)
    if X_test.shape[1] != ORAL_INPUT_DIM:
        print(f"Input dim={X_test.shape[1]} (checkpoint expects {ORAL_INPUT_DIM}). "
              f"Projecting with oral PCA...")
        X_test = project_with_oral_pca(X_test, oral_pca_path)
        print(f"✓ Projected to {X_test.shape[1]}D\n")

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(X_test)  # Shape: (N, 1) - regression output
        raw_preds = logits.squeeze(-1).cpu().numpy()  # (N,)
        # Convert to binary by thresholding at 0.5
        pred = (raw_preds > 0.5).astype(int)
        true = y_test.cpu().numpy()

    print("✓ Inference complete\n")

    # Compute metrics
    print("=" * 60)
    print("ZERO-SHOT EVALUATION: Myeloma on Oral Cancer Model")
    print("=" * 60)
    print(f"Oral Checkpoint:       {oral_checkpoint_path}")
    print(f"Oral Epoch/Val-PCC:    {epoch} / {val_pearson}")
    print(f"Myeloma Test Samples:  {len(true)}")
    print(f"Class 1 / Class 0:     {true.sum()} / {len(true) - true.sum()}")
    print("-" * 60)

    # Accuracy
    acc = (pred == true).mean()
    print(f"Accuracy:              {acc:.4f}")

    # RMSE on raw predictions
    rmse = np.sqrt(np.mean((raw_preds - true.astype(np.float64)) ** 2))
    print(f"RMSE (raw pred):       {rmse:.4f}")

    # Pearson correlation
    if len(np.unique(true)) > 1 and np.std(raw_preds) > 1e-8:
        pcc, pval = pearsonr(raw_preds, true.astype(np.float64))
        print(f"Pearson r:             {pcc:.4f}  (p={pval:.4e})")
    else:
        print(f"Pearson r:             NaN (constant prediction or single class)")

    print("-" * 60)
    print(f"Predictions (per-sample):")
    for i, (name, pred_label, raw_pred, true_label) in enumerate(
        zip(sample_names, pred, raw_preds, true)
    ):
        print(f"  {i+1}. {name:20s} | Pred={pred_label} Raw={raw_pred:+.4f} | True={true_label}")
    print("=" * 60)


if __name__ == "__main__":
    main()
