"""
Preprocess colorectal cancer Hi-C data into genome-wide contact profile feature vectors.

Steps:
  1. Read each sparse Hi-C matrix (bin1, bin2, contact_freq at 40kb resolution)
  2. Compute genome-wide row-sum contact profile → log1p transformed
  3. Align each profile to ORAL_INPUT_DIM=77405 (trim or zero-pad) so features
     match the pretrained oral cancer HiCTransformer's input dimension exactly
  4. Z-score normalize across samples (per bin)
  5. Label: tumor=1, normal=0
  6. 80/20 stratified split → save as .pt files with long-type labels
"""

import os
import gzip
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Must match the oral cancer model's input dimension
ORAL_INPUT_DIM = 77405

NORMAL_FILES = [
    "GSM3930257_BRD3162N-sb_40000_iced.matrix.gz",
    "GSM3930258_BRD3170N-sb_40000_iced.matrix.gz",
    "GSM3930260_BRD3179N_40000_iced.matrix.gz",
    "GSM3930262_BRD3187N_40000_iced.matrix.gz",
    "GSM3930263_BRD3328N_40000_iced.matrix.gz",
    "GSM3930264_BRD3409N-sb_40000_iced.matrix.gz",
    "GSM3930266_BRD3462N-sb_40000_iced.matrix.gz",
]

TUMOR_FILES = [
    "GSM3930256_BRD3162_40000_iced.matrix.gz",
    "GSM3930259_BRD3179_40000_iced.matrix.gz",
    "GSM3930261_BRD3187_40000_iced.matrix.gz",
    "GSM3930286_MGH1904_40000_iced.matrix.gz",
    "GSM3930289_MGH2834_40000_iced.matrix.gz",
    "GSM3930290_MGH5328_40000_iced.matrix.gz",
    "GSM3930291_MGH8416_40000_iced.matrix.gz",
    "GSM4513970_BRD3378_40000_iced.matrix.gz",
    "GSM4513974_BRD3462_40000_iced.matrix.gz",
    "GSM4513992_MGH3535_40000_iced.matrix.gz",
]


def extract_genome_rowsum_profile(filepath, target_dim):
    """
    Read sparse Hi-C matrix → genome-wide row-sum contact profile, log1p-transformed.
    Trims to target_dim if longer, zero-pads if shorter, so all samples have
    exactly target_dim features matching the pretrained oral cancer model.
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
            profile[b2] += val
    profile = np.log1p(profile).astype(np.float32)

    if len(profile) >= target_dim:
        profile = profile[:target_dim]
    else:
        profile = np.pad(profile, (0, target_dim - len(profile)))
    return profile


def preprocess(hic_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    features, labels, names = [], [], []

    all_files = [(f, 0) for f in NORMAL_FILES] + [(f, 1) for f in TUMOR_FILES]
    for fname, label in all_files:
        fpath = os.path.join(hic_dir, fname)
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found, skipping")
            continue
        print(f"Processing {fname} (label={'tumor' if label else 'normal'}) ...")
        profile = extract_genome_rowsum_profile(fpath, ORAL_INPUT_DIM)
        features.append(profile)
        labels.append(label)
        names.append(fname)

    features = np.array(features, dtype=np.float32)  # (N, ORAL_INPUT_DIM)
    labels   = np.array(labels,   dtype=np.int64)     # (N,)

    # z-score normalize across samples (per bin), consistent with oral cancer preprocessing
    mean = features.mean(axis=0)
    std  = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    # stratified 80/20 split
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        features, labels, names, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTrain: {len(y_train)} samples ({sum(y_train)} tumor, {len(y_train)-sum(y_train)} normal)")
    print(f"Test:  {len(y_test)} samples ({sum(y_test)} tumor, {len(y_test)-sum(y_test)} normal)")
    print(f"Train samples: {names_train}")
    print(f"Test samples:  {names_test}")
    print(f"Feature shape: {features.shape}  (input_dim={ORAL_INPUT_DIM})")

    # labels saved as torch.long — required by CrossEntropyLoss
    torch.save(torch.tensor(X_train, dtype=torch.float32),
               os.path.join(out_dir, "colorectal_train_data.pt"))
    torch.save(torch.tensor(y_train, dtype=torch.long),
               os.path.join(out_dir, "colorectal_train_labels.pt"))
    torch.save(torch.tensor(X_test,  dtype=torch.float32),
               os.path.join(out_dir, "colorectal_test_data.pt"))
    torch.save(torch.tensor(y_test,  dtype=torch.long),
               os.path.join(out_dir, "colorectal_test_labels.pt"))
    torch.save({"mean": torch.tensor(mean), "std": torch.tensor(std)},
               os.path.join(out_dir, "colorectal_norm_stats.pt"))

    # PCA: fit on train, apply to test
    N_COMPONENTS = min(100, len(X_train) - 1)
    print(f"\nFitting PCA (n_components={N_COMPONENTS}) on training data...")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)
    print(f"PCA train shape: {X_train_pca.shape}")
    print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    torch.save(torch.tensor(X_train_pca, dtype=torch.float32),
               os.path.join(out_dir, "colorectal_train_pca.pt"))
    torch.save(torch.tensor(X_test_pca,  dtype=torch.float32),
               os.path.join(out_dir, "colorectal_test_pca.pt"))
    torch.save({
        "components":   pca.components_,
        "mean":         pca.mean_,
        "n_components": pca.n_components_,
    }, os.path.join(out_dir, "colorectal_pca_model.pt"))

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess colorectal cancer Hi-C data with PCA.")
    parser.add_argument("--hic_dir", required=True,
                        help="Directory containing *_40000_iced.matrix.gz files")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to save processed .pt files")
    args = parser.parse_args()
    preprocess(hic_dir=args.hic_dir, out_dir=args.out_dir)

