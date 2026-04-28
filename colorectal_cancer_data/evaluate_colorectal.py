"""
Evaluate a fine-tuned HiCTransformer on the colorectal cancer test set.
Metrics (per proposal): Accuracy, RMSE, Pearson Correlation Coefficient.

Loads a flat state dict saved by ctf_transformer_pipeline_colorectal.py
(colorectal_cancer_freeze{0|1}_best.pt).
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

# ── Architecture constants — must match ctf_transformer_pipeline_colorectal.py ─
ORAL_INPUT_DIM     = 77405
D_MODEL            = 128
NHEAD              = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD    = 256
DROPOUT            = 0.1


class HiCTransformer(nn.Module):
    """
    Transformer encoder for Hi-C contact data.
    Copied from oral_cancer_data/train_oral.py — no path dependency needed.
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, dropout, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)

def compute_roc(pred_probs, true_labels):
    """Return (fprs, tprs, auc) arrays for a single set of predictions."""
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
    fprs.append(1.0); tprs.append(1.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    roc_auc = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, roc_auc


def plot_roc_curves(curves, out_path="colorectal_roc_curve.png"):
    """
    curves: list of (fprs, tprs, auc, label) — one entry per freeze.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Distinct colours for up to ~8 freeze runs.
    palette = ["#3A86FF", "#FF006E", "#FB5607", "#8338EC",
               "#06D6A0", "#FFB703", "#023047", "#E63946"]

    plt.figure(figsize=(6, 5))
    for i, (fprs, tprs, auc, label) in enumerate(curves):
        color = palette[i % len(palette)]
        plt.plot(fprs, tprs, color=color, lw=2, label=f"{label}  AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Colorectal Cancer Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")


def _freeze_label(model_path):
    """Extract a human-readable freeze label from a model path, e.g. 'Freeze 0'."""
    import os, re
    name = os.path.basename(model_path)
    m = re.search(r"freeze(\d+)", name, re.IGNORECASE)
    if m:
        return f"Freeze {m.group(1)}"
    # Fall back to the filename stem
    return os.path.splitext(name)[0]


def evaluate_model(model_path, X_test, y_test, device):
    """Load model, run inference, print metrics, return (pred_tumor_prob, true_labels)."""
    input_dim = X_test.shape[-1]

    model = HiCTransformer(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits          = model(X_test)
        probs           = F.softmax(logits, dim=1)
        pred_tumor_prob = probs[:, 1].cpu().numpy()       # P(tumor)
        pred_labels     = logits.argmax(dim=1).cpu().numpy()
        true_labels     = y_test.cpu().numpy()

    # 1. Accuracy
    acc = (pred_labels == true_labels).mean()

    # 2. RMSE between predicted P(tumor) and true binary label
    rmse = np.sqrt(np.mean((pred_tumor_prob - true_labels.astype(np.float64)) ** 2))

    # 3. Pearson correlation between P(tumor) and true label
    if len(np.unique(true_labels)) > 1 and np.std(pred_tumor_prob) > 1e-8:
        pcc, pval = pearsonr(pred_tumor_prob, true_labels.astype(np.float64))
    else:
        pcc, pval = float("nan"), float("nan")

    label = _freeze_label(model_path)
    print("=" * 52)
    print(f"Colorectal Cancer Test Set Evaluation  [{label}]")
    print("=" * 52)
    print(f"  Model:         {model_path}")
    print(f"  Samples:       {len(true_labels)}")
    print(f"  Tumor/Normal:  {true_labels.sum()} / {len(true_labels) - true_labels.sum()}")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  RMSE:          {rmse:.4f}")
    if not np.isnan(pcc):
        print(f"  Pearson r:     {pcc:.4f}  (p={pval:.4e})")
    else:
        print(f"  Pearson r:     NaN (constant predictions or single class)")
    print("=" * 52)
    print(f"\n  Predictions:  {pred_labels}")
    print(f"  True labels:  {true_labels}")
    print(f"  P(tumor):     {np.round(pred_tumor_prob, 4)}\n")

    return pred_tumor_prob, true_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned HiCTransformer on colorectal cancer test set"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned flat state dict (.pt)")
    parser.add_argument("--test_data",  type=str, required=True,
                        help="Path to colorectal_test_data.pt")
    parser.add_argument("--test_labels", type=str, required=True,
                        help="Path to colorectal_test_labels.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test = torch.load(args.test_data,   map_location=device).float()
    y_test = torch.load(args.test_labels, map_location=device).long()

    pred_probs, true_labels = evaluate_model(args.model_path, X_test, y_test, device)
    fprs, tprs, auc = compute_roc(pred_probs, true_labels)
    label = _freeze_label(args.model_path)
    out_path = f"colorectal_roc_curve_{label.lower().replace(' ', '')}.png"
    plot_roc_curves([(fprs, tprs, auc, label)], out_path=out_path)
