"""
Generate evaluation plots for both freeze options of the fine-tuned HiCTransformer.
Produces:
  - colorectal_evaluation_plots.png  (2-column figure: per-sample probs + metrics)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = "/athena/angsd/scratch/ssl4003/deep-learning/colorectal_cancer_data"
CHECKPOINT = {
    0: f"{BASE}/checkpoints/colorectal_cancer_freeze0_best.pt",
    1: f"{BASE}/checkpoints/colorectal_cancer_freeze1_best.pt",
}
TEST_DATA   = f"{BASE}/processed/colorectal_test_pca.pt"
TEST_LABELS = f"{BASE}/processed/colorectal_test_labels.pt"
OUT_FILE    = {0: f"{BASE}/colorectal_evaluation_freeze0.png",
               1: f"{BASE}/colorectal_evaluation_freeze1.png"}

# ── architecture ───────────────────────────────────────────────────────────────
D_MODEL            = 128
NHEAD              = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD    = 256
DROPOUT            = 0.1


class HiCTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 dim_feedforward, dropout, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
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


def run_model(freeze_opt, X, y, device):
    input_dim = X.shape[-1]
    model = HiCTransformer(input_dim, D_MODEL, NHEAD, NUM_ENCODER_LAYERS,
                           DIM_FEEDFORWARD, DROPOUT, output_dim=2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT[freeze_opt], map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs  = F.softmax(logits, dim=1)
        p_tumor = probs[:, 1].cpu().numpy()
        preds   = logits.argmax(dim=1).cpu().numpy()
    return p_tumor, preds


def pearsonr_np(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return (xm * ym).sum() / denom if denom > 1e-12 else float("nan")


def metrics(p_tumor, preds, true):
    acc  = (preds == true).mean()
    rmse = np.sqrt(np.mean((p_tumor - true.astype(float))**2))
    pcc  = pearsonr_np(p_tumor, true.astype(float))
    return acc, rmse, pcc


# ── main ───────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.load(TEST_DATA,   map_location=device).float()
y = torch.load(TEST_LABELS, map_location=device).long()
true = y.cpu().numpy()
n    = len(true)
xs   = np.arange(n)

results = {}
for opt in [0, 1]:
    p, preds = run_model(opt, X, y, device)
    acc, rmse, pcc = metrics(p, preds, true)
    results[opt] = dict(p_tumor=p, preds=preds, acc=acc, rmse=rmse, pcc=pcc)
    print(f"freeze_option={opt}  acc={acc:.4f}  rmse={rmse:.4f}  pearson={pcc:.4f}")

# ── figure layout ──────────────────────────────────────────────────────────────
TITLES = {
    0: "Freeze Option 0\n(encoder+input_proj frozen, norm+head trained)",
    1: "Freeze Option 1\n(all frozen except classification head)",
}
COLORS_TRUE  = {0: "#4878CF", 1: "#D65F5F"}   # blue=normal, red=tumor
COLORS_PRED  = {0: "#96B4E8", 1: "#EBA59A"}

for opt in [0, 1]:
    r = results[opt]
    p = r["p_tumor"]
    preds = r["preds"]

    fig = plt.figure(figsize=(7, 13))
    fig.suptitle(f"Colorectal Cancer Fine-Tuning Evaluation\n{TITLES[opt]}",
                 fontsize=12, fontweight="bold", y=0.99)
    gs = GridSpec(4, 1, figure=fig, hspace=0.6)

    # ── Row 0: per-sample P(tumor) bar chart ──────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    bar_colors = [COLORS_TRUE[t] for t in true]
    bars = ax0.bar(xs, p, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
    # mark wrong predictions with an X
    for i, (pred, tl) in enumerate(zip(preds, true)):
        if pred != tl:
            ax0.text(i, p[i] + 0.02, "✗", ha="center", va="bottom",
                     fontsize=12, color="black", fontweight="bold")
        else:
            ax0.text(i, p[i] + 0.02, "✓", ha="center", va="bottom",
                     fontsize=11, color="green", fontweight="bold")
    ax0.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision boundary")
    ax0.set_ylim(0, 1.1)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([f"S{i+1}\n({'T' if t else 'N'})" for i, t in enumerate(true)])
    ax0.set_ylabel("P(tumor)")
    ax0.set_title("Per-Sample Predictions", fontsize=10)
    ax0.set_xlabel("Sample (T=tumor, N=normal)")
    patch_t = mpatches.Patch(color=COLORS_TRUE[1], label="True: Tumor")
    patch_n = mpatches.Patch(color=COLORS_TRUE[0], label="True: Normal")
    ax0.legend(handles=[patch_t, patch_n], fontsize=7, loc="upper right")
    ax0.grid(axis="y", alpha=0.4, zorder=0)

    # ── Row 1: confusion matrix ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    cm = np.zeros((2, 2), dtype=int)
    for t, p_label in zip(true, preds):
        cm[t, p_label] += 1
    im = ax1.imshow(cm, cmap="Blues", vmin=0, vmax=max(2, cm.max()))
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Pred: Normal", "Pred: Tumor"])
    ax1.set_yticklabels(["True: Normal", "True: Tumor"])
    ax1.set_title("Confusion Matrix", fontsize=10)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # ── Row 2: metrics bar chart ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    metric_names  = ["Accuracy", "RMSE", "Pearson r"]
    metric_values = [r["acc"], r["rmse"], r["pcc"] if not np.isnan(r["pcc"]) else 0]
    bar_c = ["#3A86FF", "#FF6B6B", "#8AC926"]
    brs = ax2.bar(metric_names, metric_values, color=bar_c, edgecolor="black", linewidth=0.6)
    for bar, val in zip(brs, metric_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.1, 1.15)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Value")
    ax2.set_title("Evaluation Metrics", fontsize=10)
    ax2.grid(axis="y", alpha=0.4)

    # ── Row 3: ROC curve ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3, 0])
    if len(np.unique(true)) > 1:
        fpr, tpr, _ = roc_curve(true, r["p_tumor"])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, color="#3A86FF", lw=2,
                 label=f"AUC = {roc_auc:.3f}")
    else:
        roc_auc = float("nan")
        ax3.text(0.5, 0.5, "Single class — AUC undefined",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=9)
    ax3.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve", fontsize=10)
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(alpha=0.4)

    plt.savefig(OUT_FILE[opt], dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {OUT_FILE[opt]}")
