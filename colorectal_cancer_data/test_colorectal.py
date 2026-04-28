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
    colorectal_train_data_path = "/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/processed/colorectal_train_data.pt"
    colorectal_train_labels_path = "/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/processed/colorectal_train_labels.pt"
    colorectal_test_data_path = "/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/processed/colorectal_test_data.pt"
    colorectal_test_labels_path = "/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data/processed/colorectal_test_labels.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load oral checkpoint
    print("Loading oral cancer checkpoint (epoch 99)...")
    checkpoint = torch.load(oral_checkpoint_path, map_location=device)
    epoch = checkpoint.get("epoch", "?")
    val_pearson = checkpoint.get("val_pearson", "?")
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    
    print(f"  Checkpoint metadata: epoch={epoch}, val_pearson={val_pearson}")

    # Initialize model and load weights
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

    # Load all colorectal data (train + test)
    print("Loading all colorectal samples (train + test)...")
    X_train = torch.load(colorectal_train_data_path, map_location="cpu").float()
    y_train = torch.load(colorectal_train_labels_path, map_location="cpu").long()
    
    X_test = torch.load(colorectal_test_data_path, map_location="cpu").float()
    y_test = torch.load(colorectal_test_labels_path, map_location="cpu").long()
    
    # Concatenate
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    print(f"✓ Loaded {X_all.shape[0]} samples total ({n_train} train + {n_test} test)")
    print(f"  Features: {X_all.shape[1]}")
    print(f"  Class distribution: {y_all.sum().item()} class-1 / {len(y_all) - y_all.sum().item()} class-0\n")

    # Project if needed
    if X_all.shape[1] != ORAL_INPUT_DIM:
        print(f"Input dim={X_all.shape[1]} (checkpoint expects {ORAL_INPUT_DIM}). "
              f"Projecting with oral PCA...")
        X_all = project_with_oral_pca(X_all, oral_pca_path)
        print(f"✓ Projected to {X_all.shape[1]}D\n")

    X_all = X_all.to(device)
    y_all = y_all.to(device)

    # Run inference
    print("Running inference on all samples...")
    with torch.no_grad():
        logits = model(X_all)
        raw_preds = logits.squeeze(-1).cpu().numpy()
        pred = (raw_preds > 0.5).astype(int)
        true = y_all.cpu().numpy()

    print("✓ Inference complete\n")

    # Compute overall metrics
    print("=" * 70)
    print("ZERO-SHOT EVALUATION: All Colorectal Samples on Oral Cancer Model")
    print("=" * 70)
    print(f"Oral Checkpoint:           {oral_checkpoint_path}")
    print(f"Oral Epoch/Val-PCC:        {epoch} / {val_pearson}")
    print(f"Total Colorectal Samples:     {len(true)} (train={n_train}, test={n_test})")
    print(f"Class 1 / Class 0:         {true.sum()} / {len(true) - true.sum()}")
    print("-" * 70)

    acc = (pred == true).mean()
    print(f"Overall Accuracy:          {acc:.4f}")

    rmse = np.sqrt(np.mean((raw_preds - true.astype(np.float64)) ** 2))
    print(f"Overall RMSE:              {rmse:.4f}")

    if len(np.unique(true)) > 1 and np.std(raw_preds) > 1e-8:
        pcc, pval = pearsonr(raw_preds, true.astype(np.float64))
        print(f"Overall Pearson r:         {pcc:.4f}  (p={pval:.4e})")
    else:
        print(f"Overall Pearson r:         NaN (constant prediction or single class)")

    # Per-subset metrics
    train_mask = np.arange(len(y_all)) < n_train
    test_mask = np.arange(len(y_all)) >= n_train
    
    if train_mask.sum() > 0:
        train_acc = (pred[train_mask] == true[train_mask]).mean()
        print(f"Train Accuracy:            {train_acc:.4f} ({train_mask.sum()} samples)")
    
    if test_mask.sum() > 0:
        test_acc = (pred[test_mask] == true[test_mask]).mean()
        print(f"Test Accuracy:             {test_acc:.4f} ({test_mask.sum()} samples)")

    print("-" * 70)
    print(f"Per-Sample Predictions:")
    print()
    
    for i, (pred_label, raw_pred, true_label) in enumerate(
        zip(pred, raw_preds, true)
    ):
        subset = "TRAIN" if i < n_train else "TEST "
        match = "✓" if pred_label == true_label else "✗"
        print(f"  {i+1}. [{subset}] {match} | Pred={pred_label} Raw={raw_pred:+.4f} | True={true_label}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
