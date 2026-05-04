"""
Preprocess oral cancer Hi-C data into CTG feature vectors for transformer training.

Steps:
  1. Read each sparse Hi-C matrix (bin1, bin2, contact_freq at 40kb resolution)
  2. Extract intra-chromosomal contacts for chr1 (bins 0–6231, matching input_dim=6232)
  3. Compute per-bin contact profile (row sums of chr1 contact matrix) → 6232-dim vector
  4. Log-transform and z-score normalize
  5. Label: tumor (OACC/OSCC) = 1, normal (E/G) = 0
  6. Save ALL as training data (no val split per proposal — tested on other cancer types)
"""

print("Loading libraries and defining configuration...")
from sklearn.decomposition import PCA
import os
print("Current working directory:", os.getcwd())
import glob
print("Glob module imported successfully.")
import gzip
print("Gzip module imported successfully.")
import numpy as np
print("Numpy version:", np.__version__)
import torch
print("PyTorch version:", torch.__version__)
import torch.nn as nn
print("Torch NN module imported successfully.")
from torch.utils.data import DataLoader, TensorDataset, random_split
print("Torch Data utilities imported successfully.")
from scipy.stats import pearsonr
print("Scipy stats imported successfully.")

import datetime
import time
# Config
BASE_DIR = os.getcwd()
CHR1_BINS = 6232  # chr1 at 40kb resolution (hg19: 249,250,621 bp / 40,000)
TRAIN_SPLIT       = 0.8
BATCH_SIZE        = 32
NUM_EPOCHS        = 150
LEARNING_RATE     = 1e-4
D_MODEL           = 128
NHEAD             = 8
NUM_ENCODER_LAYERS= 4
DIM_FEEDFORWARD   = 256
DROPOUT           = 0.1
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
hic_dir = os.path.join(BASE_DIR, "../HiC")
out_dir = os.path.join(BASE_DIR, f"processed_{RUN_ID}")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"checkpoints_{RUN_ID}")
# CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def extract_chr1_profile(filepath):
    """Read sparse Hi-C matrix and return chr1 row-sum contact profile (6232-dim)."""
    profile = np.zeros(CHR1_BINS, dtype=np.float64)
    with gzip.open(filepath, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            b1, b2, val = int(parts[0]), int(parts[1]), float(parts[2])
            # keep only intra-chr1 contacts
            if b1 < CHR1_BINS and b2 < CHR1_BINS:
                profile[b1] += val
                if b1 != b2:
                    profile[b2] += val
    # log1p transform to reduce skew from high-contact regions
    profile = np.log1p(profile)
    return profile

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

def get_label(filename):
    """Assign label based on filename prefix: E*/G* = normal (0), OACC*/OSCC* = tumor (1)."""
    basename = os.path.basename(filename)
    if basename.startswith("E") or basename.startswith("G"):
        return 0
    elif basename.startswith("OACC") or basename.startswith("OSCC"):
        return 1
    else:

        raise ValueError(f"Unknown sample prefix: {basename}")


def preprocess(hic_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning Hi-C files in {hic_dir} ...")
    hic_files = sorted(glob.glob(os.path.join(hic_dir, "*_40000_iced.matrix.gz")))
    print(f"Found {len(hic_files)} Hi-C files")

    features = []
    labels = []
    names = []

    for fpath in hic_files:
        fname = os.path.basename(fpath)
        label = get_label(fname)
        print(f"Processing {fname} (label={'tumor' if label else 'normal'}) ...")
        #profile = extract_chr1_profile(fpath)
        profile = extract_genome_rowsum_profile(fpath)
        features.append(profile)
        labels.append(label)
        names.append(fname)

    features = np.array(features, dtype=np.float32)  # (N, 6232)
    labels = np.array(labels, dtype=np.int64)          # (N,)

    # z-score normalize across samples (per feature/bin)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    n_tumor = sum(labels)
    n_normal = len(labels) - n_tumor
    #print(f"\nTotal: {len(labels)} samples ({n_tumor} tumor, {n_normal} normal)")
    #print("All samples used for training (no val split — model tested on other cancer types)")

    # save as training tensors
    torch.save(torch.tensor(features), os.path.join(out_dir, "oral_train_data.pt"))
    torch.save(torch.tensor(labels), os.path.join(out_dir, "oral_train_labels.pt"))

    # save normalization stats (needed to normalize other cancer types consistently)
    torch.save({"mean": torch.tensor(mean), "std": torch.tensor(std)},
               os.path.join(out_dir, "oral_norm_stats.pt"))

    print(f"Saved to {out_dir}/")

    torch_data = torch.tensor(features, dtype=torch.float32)
    torch_labels = torch.tensor(labels, dtype=torch.float32)
    print(f"Data shape: {torch_data.shape} | dtype: {torch_data.dtype}")
    print(f"Labels shape: {torch_labels.shape} | dtype: {torch_labels.dtype}")
    return torch_data, torch_labels

print("Starting preprocessing of oral cancer Hi-C data...")
DATA_PATH = os.path.join(out_dir, "oral_train_data.pt")
LABELS_PATH = os.path.join(out_dir, "oral_train_labels.pt")

# if not os.path.exists(DATA_PATH):
#     data, labels = preprocess(hic_dir, out_dir)
# else:
print("Loading existing processed data...")
data = torch.load('/athena/angsd/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_001034/oral_train_data.pt')
labels = torch.load('/athena/angsd/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_001034/oral_train_labels.pt')


print(f"Loading data from {DATA_PATH} and {LABELS_PATH}...")
# data   = torch.load(DATA_PATH,   map_location="cpu")
# labels = torch.load(LABELS_PATH, map_location="cpu")

print(f"Data shape:   {data.shape}   | dtype: {data.dtype}")
print(f"Labels shape: {labels.shape} | dtype: {labels.dtype}")

# Ensure float tensors
data   = data.float()
labels = labels.float()


# ~~adding pca~~
# Ensure float tensors
data   = data.float()
labels = labels.float()

# Split first
print("splitting")
dataset    = TensorDataset(data, labels)
n_total    = len(dataset)
n_train    = int(n_total * TRAIN_SPLIT)
n_val      = n_total - n_train
train_set, val_set = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

# Extract tensors from subsets
print("extract tensors from subsets")
X_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
y_train = torch.stack([train_set[i][1] for i in range(len(train_set))])

X_val = torch.stack([val_set[i][0] for i in range(len(val_set))])
y_val = torch.stack([val_set[i][1] for i in range(len(val_set))])

# Fit PCA on training data only
print("fit pca")
N_COMPONENTS = min(100, X_train.shape[0] - 1)

t0 = time.time()
print("Starting PCA...", flush=True)

pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=42)
X_train_pca = pca.fit_transform(X_train.numpy())
X_val_pca   = pca.transform(X_val.numpy())

print(f"PCA finished in {time.time() - t0:.2f} seconds", flush=True)

print("Original train shape:", X_train.shape)
print("PCA train shape:", X_train_pca.shape)
print("Explained variance ratio sum:", pca.explained_variance_ratio_.sum())

# Convert back to torch tensors
X_train_pca = torch.tensor(X_train_pca, dtype=torch.float32)
X_val_pca   = torch.tensor(X_val_pca, dtype=torch.float32)

# Save PCA artifacts
PCA_TRAIN_PATH = os.path.join(out_dir, "X_train_pca.pt")
PCA_VAL_PATH   = os.path.join(out_dir, "X_val_pca.pt")

torch.save(X_train_pca, PCA_TRAIN_PATH)
torch.save(X_val_pca,   PCA_VAL_PATH)

PCA_MODEL_PATH = os.path.join(out_dir, "pca_model.pt")

torch.save({
    "components": pca.components_,
    "mean": pca.mean_,
    "n_components": pca.n_components_
}, PCA_MODEL_PATH)


# New datasets/loaders
train_set = TensorDataset(X_train_pca, y_train.float())
val_set   = TensorDataset(X_val_pca, y_val.float())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nDataset split: {n_train} train / {n_val} val samples")
print(pca.explained_variance_ratio_.sum())

# # Dataset & Split 
# dataset    = TensorDataset(data, labels)
# n_total    = len(dataset)
# n_train    = int(n_total * TRAIN_SPLIT)
# n_val      = n_total - n_train
# train_set, val_set = random_split(dataset, [n_train, n_val],
#                                    generator=torch.Generator().manual_seed(42))

# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
# val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# print(f"\nDataset split: {n_train} train / {n_val} val samples")

# Model
class HiCTransformer(nn.Module):
    """
    Transformer encoder for regression on Hi-C contact data.
    Input:  (batch, seq_len, feature_dim)  or  (batch, feature_dim)
    Output: (batch, output_dim)
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
        self.encoder    = nn.TransformerEncoder(encoder_layer,
                                                num_layers=num_encoder_layers)
        self.norm       = nn.LayerNorm(d_model)
        self.head       = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)          # treat flat feature as seq_len=1
        x = self.input_proj(x)          # â†’ (batch, seq_len, d_model)
        x = self.encoder(x)             # â†’ (batch, seq_len, d_model)
        x = self.norm(x.mean(dim=1))    # mean-pool over sequence â†’ (batch, d_model)
        return self.head(x)             # â†’ (batch, output_dim)


# Infer dims from data
input_dim = N_COMPONENTS
output_dim = labels.shape[-1] if labels.dim() > 1 else 1

model = HiCTransformer(
    input_dim=input_dim,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    output_dim=output_dim
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: HiCTransformer | Parameters: {n_params:,} | Device: {DEVICE}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)
criterion = nn.MSELoss()

# Metrics
def compute_pearson(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute Pearson correlation; handles flat and multi-output tensors."""
    preds   = preds.flatten()
    targets = targets.flatten()
    if np.std(preds) < 1e-8 or np.std(targets) < 1e-8:
        return 0.0
    r, _ = pearsonr(preds, targets)
    return float(r)

# Training Loop
best_val_pearson = -np.inf
history = {"train_loss": [], "val_mse": [], "val_pearson": []}

print("\n" + "="*65)
print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Val Pearson':>12}")
print("="*65)

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    train_loss_sum = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        if y_batch.dim() == 1:
            y_batch = y_batch.unsqueeze(-1)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss_sum += loss.item() * X_batch.size(0)

    train_mse = train_loss_sum / n_train

    # Validate
    model.eval()
    val_loss_sum = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(-1)
            preds = model(X_batch)
            val_loss_sum += criterion(preds, y_batch).item() * X_batch.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    val_mse     = val_loss_sum / n_val
    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_pearson = compute_pearson(all_preds, all_targets)

    scheduler.step(val_mse)

    history["train_loss"].append(train_mse)
    history["val_mse"].append(val_mse)
    history["val_pearson"].append(val_pearson)

    print(f"{epoch:>6}  {train_mse:>10.5f}  {val_mse:>10.5f}  {val_pearson:>12.4f}")

    # Checkpoint
    if val_pearson > best_val_pearson:
        best_val_pearson = val_pearson
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_pearson":     val_pearson,
            "val_mse":         val_mse,
        }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

print("="*65)
print(f"\nTraining complete. Best Val Pearson: {best_val_pearson:.4f}")
print(f"Best checkpoint saved to: {CHECKPOINT_DIR}/best_model.pt")
