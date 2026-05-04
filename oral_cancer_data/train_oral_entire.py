#!/usr/bin/env python3
"""
Train a Transformer on oral cancer Hi-C data from multiple gzipped .matrix.gz files.
Uses a genome-wide row-sum profile (all chromosomes) as features.
Applies log1p + z-score normalization.
"""

import os
import glob
import gzip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.stats import pearsonr

HIC_DIR = "HiC"
OUT_DIR = "./processed"
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_label(filename):
    """Assign label based on filename prefix: E*/G* = normal (0), OACC*/OSCC* = tumor (1)."""
    basename = os.path.basename(filename)
    if basename.startswith("E") or basename.startswith("G"):
        return 0
    elif basename.startswith("OACC") or basename.startswith("OSCC"):
        return 1
    else:
        raise ValueError(f"Unknown sample prefix: {basename}")


def extract_genome_rowsum_profile(filepath):
    """
    Read a gzipped sparse Hi-C matrix (bin1, bin2, value) across all chromosomes
    and return a genome-wide row-sum contact profile.
    Matrix size is inferred from the max bin index found in the file.
    Applies log1p to reduce skew from high-contact regions.
    """
    rows, cols, vals = [], [], []
    with gzip.open(filepath, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            b1, b2, val = int(parts[0]), int(parts[1]), float(parts[2])
            rows.append(b1)
            cols.append(b2)
            vals.append(val)

    n_bins = max(max(rows), max(cols)) + 1
    profile = np.zeros(n_bins, dtype=np.float64)
    for b1, b2, val in zip(rows, cols, vals):
        profile[b1] += val
        if b1 != b2:
            profile[b2] += val  # symmetric contribution

    profile = np.log1p(profile)
    return profile.astype(np.float32)


def preprocess(hic_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Scanning Hi-C files in {hic_dir} ...", flush=True)
    hic_files = sorted(glob.glob(os.path.join(hic_dir, "*_40000_iced.matrix.gz")))
    print(f"Found {len(hic_files)} Hi-C files", flush=True)

    features = []
    labels = []

    for fpath in hic_files:
        fname = os.path.basename(fpath)
        label = get_label(fname)
        print(f"Processing {fname} (label={'tumor' if label else 'normal'}) ...", flush=True)
        profile = extract_genome_rowsum_profile(fpath)
        features.append(profile)
        labels.append(label)

    if len(features) == 0:
        raise ValueError(f"No files matched pattern in {hic_dir}")

    # Trim all profiles to the shortest common length to allow stacking
    min_bins = min(len(p) for p in features)
    print(f"Trimming all profiles to {min_bins} bins (genome-wide).", flush=True)
    features = np.stack([p[:min_bins] for p in features], axis=0).astype(np.float32)  # (N, min_bins)
    labels   = np.array(labels, dtype=np.int64)                                        # (N,)

    print(f"Feature matrix shape: {features.shape}", flush=True)

    # z-score normalize across samples (per bin)
    mean = features.mean(axis=0)
    std  = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    n_tumor  = int(labels.sum())
    n_normal = int(len(labels) - n_tumor)
    print(f"Total: {len(labels)} samples ({n_tumor} tumor, {n_normal} normal)", flush=True)

    torch.save(torch.tensor(features, dtype=torch.float32), os.path.join(out_dir, "oral_train_data.pt"))
    torch.save(torch.tensor(labels,   dtype=torch.int64),   os.path.join(out_dir, "oral_train_labels.pt"))
    torch.save({"mean": torch.tensor(mean), "std": torch.tensor(std)}, os.path.join(out_dir, "oral_norm_stats.pt"))
    print(f"Saved tensors to {out_dir}/", flush=True)

    torch_data   = torch.tensor(features, dtype=torch.float32)
    torch_labels = torch.tensor(labels,   dtype=torch.float32)
    print(f"Data shape:   {torch_data.shape}   | dtype: {torch_data.dtype}", flush=True)
    print(f"Labels shape: {torch_labels.shape} | dtype: {torch_labels.dtype}", flush=True)
    return torch_data, torch_labels


class HiCTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
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


def compute_pearson(preds, targets):
    preds   = preds.flatten()
    targets = targets.flatten()
    if np.std(preds) < 1e-8 or np.std(targets) < 1e-8:
        return 0.0
    return float(pearsonr(preds, targets)[0])


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data, labels = preprocess(HIC_DIR, OUT_DIR)

    dataset = TensorDataset(data, labels)
    n_total = len(dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val   = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Dataset split: {n_train} train / {n_val} val samples", flush=True)

    input_dim  = data.shape[-1]
    output_dim = 1
    model = HiCTransformer(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=output_dim,
    ).to(DEVICE)

    print(f"Model on {DEVICE} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_pearson = -np.inf
    print("\n" + "=" * 65, flush=True)
    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Val Pearson':>12}", flush=True)
    print("=" * 65, flush=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).float().unsqueeze(-1)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * X_batch.size(0)

        train_mse = train_loss_sum / n_train

        model.eval()
        val_loss_sum = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE).float().unsqueeze(-1)
                preds   = model(X_batch)
                val_loss_sum += criterion(preds, y_batch).item() * X_batch.size(0)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_mse     = val_loss_sum / n_val
        all_preds   = np.concatenate(all_preds,   axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        val_pearson = compute_pearson(all_preds, all_targets)
        scheduler.step(val_mse)

        print(f"{epoch:6d}  {train_mse:10.5f}  {val_mse:10.5f}  {val_pearson:12.4f}", flush=True)

        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_pearson":     val_pearson,
                "val_mse":         val_mse,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    print("=" * 65, flush=True)
    print(f"Training complete. Best Val Pearson: {best_val_pearson:.4f}", flush=True)
    print(f"Best checkpoint saved to: {CHECKPOINT_DIR}/best_model.pt", flush=True)


if __name__ == "__main__":
    main()