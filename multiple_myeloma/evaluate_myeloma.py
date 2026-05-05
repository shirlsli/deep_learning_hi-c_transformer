"""
Evaluate fine-tuned HiCTransformer on multiple myeloma test set.
Metrics: Accuracy, RMSE, Pearson r.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

# Oral checkpoint was trained on PCA features with 76 dimensions.
ORAL_INPUT_DIM = 76
DEFAULT_ORAL_PCA_PATH = "/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_133858/pca_model.pt"
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


def evaluate(model_path, test_data_path, test_labels_path, pca_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HiCTransformer(
        input_dim=ORAL_INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_test = torch.load(test_data_path, map_location="cpu").float()
    y_test = torch.load(test_labels_path, map_location=device).long()

    if X_test.shape[1] != ORAL_INPUT_DIM:
        print(
            f"Input dim={X_test.shape[1]} (checkpoint expects {ORAL_INPUT_DIM}). "
            f"Projecting with oral PCA: {pca_model_path}"
        )
        X_test = project_with_oral_pca(X_test, pca_model_path)

    X_test = X_test.to(device)

    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1)
        p1 = probs[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        true = y_test.cpu().numpy()

    acc = (pred == true).mean()
    rmse = np.sqrt(np.mean((p1 - true.astype(np.float64)) ** 2))
    if len(np.unique(true)) > 1 and np.std(p1) > 1e-8:
        pcc, pval = pearsonr(p1, true.astype(np.float64))
    else:
        pcc, pval = float("nan"), float("nan")

    print("=" * 52)
    print("Multiple Myeloma Test Set Evaluation")
    print("=" * 52)
    print(f"  Model:         {model_path}")
    print(f"  Samples:       {len(true)}")
    print(f"  Class1/Class0: {true.sum()} / {len(true) - true.sum()}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  RMSE:          {rmse:.4f}")
    if not np.isnan(pcc):
        print(f"  Pearson r:     {pcc:.4f}  (p={pval:.4e})")
    else:
        print("  Pearson r:     NaN (constant prediction or single class)")
    print("=" * 52)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on myeloma test set")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--test_labels", type=str, required=True)
    parser.add_argument("--pca_model_path", type=str, default=DEFAULT_ORAL_PCA_PATH)
    args = parser.parse_args()
    evaluate(args.model_path, args.test_data, args.test_labels, args.pca_model_path)
