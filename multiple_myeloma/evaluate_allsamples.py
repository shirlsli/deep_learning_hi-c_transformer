#!/usr/bin/env python3
"""
Evaluate fine-tuned checkpoints (trained on all 4 samples) on all 4 myeloma samples.
Generates ROC curves.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORAL_INPUT_DIM = 76
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
DEFAULT_ORAL_PCA_PATH = "/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_133858/pca_model.pt"


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

    device = data_tensor.device
    components = components.to(device)
    mean = mean.to(device)

    if data_tensor.shape[1] != mean.shape[0]:
        raise ValueError(f"Feature dim {data_tensor.shape[1]} does not match oral PCA input dim {mean.shape[0]}")

    centered = data_tensor - mean
    projected = centered @ components.T
    return projected.float()


def plot_roc(pred_probs, true_labels, out_path):
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
    plt.title("ROC Curve — Multiple Myeloma (All Samples)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC: {out_path}")
    return roc_auc


def evaluate(model_path, all_data, all_labels, pca_model_path, freeze_opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to device
    X_all = all_data.to(device).float()
    y_all = all_labels.to(device).long()

    print(f"Loaded data: {X_all.shape}")
    print(f"Loaded labels: {y_all.shape}")

    # Project if needed
    if X_all.shape[1] != ORAL_INPUT_DIM:
        print(f"Projecting with oral PCA...")
        X_all = project_with_oral_pca(X_all, pca_model_path)
        print(f"Projected to {X_all.shape[1]}D")

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

    with torch.no_grad():
        logits = model(X_all)
        probs = F.softmax(logits, dim=1)
        pred_class1_prob = probs[:, 1].cpu().numpy()
        pred_labels = logits.argmax(dim=1).cpu().numpy()
        true_labels = y_all.cpu().numpy()

    acc = (pred_labels == true_labels).mean()
    rmse = np.sqrt(np.mean((pred_class1_prob - true_labels.astype(np.float64)) ** 2))

    if len(np.unique(true_labels)) > 1 and np.std(pred_class1_prob) > 1e-8:
        from scipy.stats import pearsonr
        pcc, pval = pearsonr(pred_class1_prob, true_labels.astype(np.float64))
    else:
        pcc, pval = float("nan"), float("nan")

    print("=" * 70)
    print(f"Freeze {freeze_opt} — All Samples Evaluation")
    print("=" * 70)
    print(f"  Model:         {model_path}")
    print(f"  Samples:       {len(true_labels)}")
    print(f"  Class 1/0:     {true_labels.sum()} / {len(true_labels) - true_labels.sum()}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  RMSE:          {rmse:.4f}")
    if not np.isnan(pcc):
        print(f"  Pearson r:     {pcc:.4f}  (p={pval:.4e})")
    else:
        print(f"  Pearson r:     NaN")
    print("=" * 70)
    print(f"  Predictions:   {pred_labels}")
    print(f"  True labels:   {true_labels}")
    print(f"  P(class1):     {np.round(pred_class1_prob, 4)}")

    # Plot ROC
    if len(np.unique(true_labels)) > 1:
        auc = plot_roc(pred_class1_prob, true_labels, f"myeloma_allsamples_roc_freeze{freeze_opt}.png")
        print(f"  AUC:           {auc:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all-samples checkpoints on all samples")
    parser.add_argument("--train_data", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_train_data.pt")
    parser.add_argument("--train_labels", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_train_labels.pt")
    parser.add_argument("--test_data", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_data.pt")
    parser.add_argument("--test_labels", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_labels.pt")
    parser.add_argument("--pca_model_path", default=DEFAULT_ORAL_PCA_PATH)
    parser.add_argument("--checkpoint_dir", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/checkpoints_all_samples")
    args = parser.parse_args()

    # Load combined data
    train_data = torch.load(args.train_data, map_location="cpu", weights_only=False).float()
    train_labels = torch.load(args.train_labels, map_location="cpu", weights_only=False).long()
    test_data = torch.load(args.test_data, map_location="cpu", weights_only=False).float()
    test_labels = torch.load(args.test_labels, map_location="cpu", weights_only=False).long()

    all_data = torch.cat([train_data, test_data], dim=0)
    all_labels = torch.cat([train_labels, test_labels], dim=0)

    print("=" * 70)
    print("All-Samples Fine-Tuned Checkpoint Evaluation")
    print("=" * 70)
    print()

    for freeze_opt in [0, 1]:
        checkpoint_path = f"{args.checkpoint_dir}/myeloma_allsamples_freeze{freeze_opt}_best.pt"
        if os.path.exists(checkpoint_path):
            evaluate(checkpoint_path, all_data, all_labels, args.pca_model_path, freeze_opt)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    print("Evaluation complete!")


if __name__ == "__main__":
    import os
    main()
