"""
Evaluate a fine-tuned HiCTransformer on the multiple myeloma test set.
Metrics: Accuracy, RMSE, Pearson Correlation Coefficient.

Loads a flat state dict saved by ctf_transformer_pipeline_myeloma.py
(myeloma_freeze{0|1}_best.pt).

Handles dimension mismatch by applying oral PCA projection if needed.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pearsonr(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    r = (xm * ym).sum() / denom if denom > 1e-12 else float("nan")
    return r, float("nan")  # p-value not computed


# ── Architecture constants — must match ctf_transformer_pipeline_myeloma.py ─
ORAL_INPUT_DIM = 76
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
DEFAULT_ORAL_PCA_PATH = "/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_133858/pca_model.pt"


class HiCTransformer(nn.Module):
    """
    Transformer encoder for Hi-C contact data.
    """

    def __init__(
        self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim
    ):
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

    # Move to same device as data
    device = data_tensor.device
    components = components.to(device)
    mean = mean.to(device)

    if data_tensor.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Feature dim {data_tensor.shape[1]} does not match oral PCA input dim {mean.shape[0]}"
        )

    centered = data_tensor - mean
    projected = centered @ components.T
    return projected.float()


def plot_predictions(pred_probs, true_labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute ROC curve manually
    thresholds = np.sort(np.unique(pred_probs))[::-1]
    fprs, tprs = [0.0], [0.0]
    n_pos = true_labels.sum()
    n_neg = len(true_labels) - n_pos
    for thresh in thresholds:
        preds = (pred_probs >= thresh).astype(int)
        tp = ((preds == 1) & (true_labels == 1)).sum()
        fp = ((preds == 1) & (true_labels == 0)).sum()
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)
    fprs.append(1.0)
    tprs.append(1.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    roc_auc = np.trapezoid(tprs, fprs)

    plt.figure(figsize=(6, 5))
    plt.plot(fprs, tprs, color="#3A86FF", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Multiple Myeloma Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig("myeloma_roc_curve.png", dpi=150)
    print("Plot saved to myeloma_roc_curve.png")


def evaluate(model_path, test_data_path, test_labels_path, pca_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    X_test = torch.load(test_data_path, map_location=device).float()
    y_test = torch.load(test_labels_path, map_location=device).long()
    input_dim = X_test.shape[-1]

    print(f"Loaded test data: {X_test.shape}")
    print(f"Loaded test labels: {y_test.shape}")
    print(f"Input dimension: {input_dim}")

    # Project if dimensions don't match expected (76)
    if input_dim != ORAL_INPUT_DIM:
        print(
            f"Input dim={input_dim} (checkpoint expects {ORAL_INPUT_DIM}). "
            f"Projecting with oral PCA: {pca_model_path}"
        )
        X_test = project_with_oral_pca(X_test, pca_model_path)
        input_dim = X_test.shape[-1]
        print(f"Projected to {input_dim}D")

    model = HiCTransformer(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        pred_tumor_prob = probs[:, 1].cpu().numpy()  # P(tumor)
        pred_labels = logits.argmax(dim=1).cpu().numpy()
        true_labels = y_test.cpu().numpy()

    # 1. Accuracy
    acc = (pred_labels == true_labels).mean()

    # 2. RMSE between predicted P(tumor) and true binary label
    rmse = np.sqrt(np.mean((pred_tumor_prob - true_labels.astype(np.float64)) ** 2))

    # 3. Pearson correlation between P(tumor) and true label
    if len(np.unique(true_labels)) > 1 and np.std(pred_tumor_prob) > 1e-8:
        pcc, pval = pearsonr(pred_tumor_prob, true_labels.astype(np.float64))
    else:
        pcc, pval = float("nan"), float("nan")

    print("=" * 60)
    print("Multiple Myeloma Test Set Evaluation")
    print("=" * 60)
    print(f"  Model:         {model_path}")
    print(f"  Samples:       {len(true_labels)}")
    print(f"  Class 1/0:     {true_labels.sum()} / {len(true_labels) - true_labels.sum()}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  RMSE:          {rmse:.4f}")
    if not np.isnan(pcc):
        print(f"  Pearson r:     {pcc:.4f}  (p={pval:.4e})")
    else:
        print(f"  Pearson r:     NaN (constant predictions or single class)")
    print("=" * 60)
    print(f"\n  Predictions:   {pred_labels}")
    print(f"  True labels:   {true_labels}")
    print(f"  P(class1):     {np.round(pred_tumor_prob, 4)}")
    if len(np.unique(true_labels)) > 1:
        plot_predictions(pred_tumor_prob, true_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned HiCTransformer on multiple myeloma test set"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned flat state dict (.pt)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_data.pt",
        help="Path to myeloma_test_data.pt",
    )
    parser.add_argument(
        "--test_labels",
        type=str,
        default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_labels.pt",
        help="Path to myeloma_test_labels.pt",
    )
    parser.add_argument(
        "--pca_model_path",
        type=str,
        default=DEFAULT_ORAL_PCA_PATH,
        help="Path to oral PCA model for projection",
    )
    args = parser.parse_args()
    evaluate(args.model_path, args.test_data, args.test_labels, args.pca_model_path)
