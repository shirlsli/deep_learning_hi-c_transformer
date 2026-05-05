"""
Preprocess multiple myeloma Hi-C data into feature vectors compatible with oral model input.

Input format in this dataset:
  - *.tar.gz archives
  - each archive contains dense per-chromosome matrices at:
    resolution_40k/cis/ice_normalization/chr*_40k_normalized_matrix.txt

Steps:
  1. Read each tar.gz sample archive
  2. For each chromosome matrix, compute row-sum profile and apply log1p
  3. Concatenate chr1..chr24 profiles into a genome-wide vector
  4. Align each vector to ORAL_INPUT_DIM=77405 (trim or zero-pad)
  5. Z-score normalize across samples (per feature)
  6. Train/test split (stratified if >=2 classes, otherwise random split)
  7. Fit PCA on train, apply to test, save tensors and PCA artifacts
"""

import argparse
import os
import re
import tarfile

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Must match oral cancer model input dimension.
ORAL_INPUT_DIM = 77405

# Default labels for this dataset (both are myeloma/cancer). You can override via --label_mode.
# label_mode=cell_line gives RPMI8226=0, U266=1 for within-myeloma discrimination.
# label_mode=all_tumor gives all samples label=1.


def chr_index_from_name(path_in_tar: str):
    match = re.search(r"chr(\d+)_40k_normalized_matrix\.txt$", path_in_tar)
    if not match:
        return None
    return int(match.group(1))


def row_sum_profile_from_dense_member(tar_obj: tarfile.TarFile, member: tarfile.TarInfo):
    """
    Read one dense chromosome matrix from tar and return log1p(row sums).
    Each row is parsed as whitespace-separated floats.
    """
    fobj = tar_obj.extractfile(member)
    if fobj is None:
        return np.array([], dtype=np.float32)

    row_sums = []
    for raw in fobj:
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        vals = np.fromstring(line, sep=" ", dtype=np.float64)
        if vals.size == 0:
            continue
        rs = vals.sum()
        if rs < 0:
            rs = 0.0
        row_sums.append(np.log1p(rs))

    return np.asarray(row_sums, dtype=np.float32)


def extract_genome_rowsum_profile_from_tar(tar_path, target_dim):
    """
    Build genome-wide profile from per-chromosome dense matrices inside one tar archive.
    """
    with tarfile.open(tar_path, "r:gz") as tar_obj:
        chr_members = []
        for member in tar_obj.getmembers():
            if not member.isfile():
                continue
            chr_idx = chr_index_from_name(member.name)
            if chr_idx is not None:
                chr_members.append((chr_idx, member))

        chr_members.sort(key=lambda x: x[0])

        pieces = []
        for _, member in chr_members:
            prof = row_sum_profile_from_dense_member(tar_obj, member)
            if prof.size > 0:
                pieces.append(prof)

    if not pieces:
        raise ValueError(f"No chr*_40k_normalized_matrix.txt entries found in {tar_path}")

    profile = np.concatenate(pieces, axis=0)

    # Align to oral model input dimension.
    if profile.shape[0] >= target_dim:
        profile = profile[:target_dim]
    else:
        profile = np.pad(profile, (0, target_dim - profile.shape[0]))

    return profile.astype(np.float32)


def infer_label_from_filename(filename, label_mode):
    name = os.path.basename(filename).upper()
    if label_mode == "all_tumor":
        return 1

    # cell_line mode
    if "RPMI8226" in name:
        return 0
    if "U266" in name:
        return 1

    raise ValueError(f"Could not infer label for file: {filename}")


def preprocess(hic_dir, out_dir, label_mode="cell_line", test_size=0.2, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    tar_files = sorted(
        [
            os.path.join(hic_dir, x)
            for x in os.listdir(hic_dir)
            if x.endswith(".tar.gz")
        ]
    )
    print(f"Found {len(tar_files)} tar.gz files in {hic_dir}")
    if len(tar_files) == 0:
        raise ValueError("No .tar.gz files found")

    features = []
    labels = []
    names = []

    for tar_path in tar_files:
        fname = os.path.basename(tar_path)
        label = infer_label_from_filename(fname, label_mode)
        print(f"Processing {fname} (label={label}) ...")
        profile = extract_genome_rowsum_profile_from_tar(tar_path, ORAL_INPUT_DIM)
        features.append(profile)
        labels.append(label)
        names.append(fname)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # z-score normalize across samples (per feature/bin).
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    unique_labels = np.unique(labels)
    n_classes = unique_labels.shape[0]
    n_samples = labels.shape[0]
    class_counts = np.bincount(labels)

    # Stratified split can fail on tiny datasets (e.g., 4 samples, test_size=0.2 -> 1 test sample).
    # Only stratify when there is enough room to place at least one sample per class in both splits.
    if isinstance(test_size, float):
        n_test = int(np.ceil(test_size * n_samples))
    else:
        n_test = int(test_size)
    n_train = n_samples - n_test

    can_stratify = (
        n_classes > 1
        and n_test >= n_classes
        and n_train >= n_classes
        and class_counts.min() >= 2
    )
    stratify_arg = labels if can_stratify else None
    if not can_stratify and n_classes > 1:
        print(
            "Warning: dataset too small for stratified split with current test_size; "
            "falling back to non-stratified split."
        )

    try:
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            features,
            labels,
            names,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )
    except ValueError as e:
        print(f"Warning: split failed with stratify={stratify_arg is not None}: {e}")
        print("Retrying with non-stratified split.")
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            features,
            labels,
            names,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    print(f"\nTrain: {len(y_train)} samples")
    print(f"Test:  {len(y_test)} samples")
    print(f"Train labels distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test labels distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print(f"Feature shape: {features.shape} (input_dim={ORAL_INPUT_DIM})")

    torch.save(torch.tensor(X_train, dtype=torch.float32), os.path.join(out_dir, "myeloma_train_data.pt"))
    torch.save(torch.tensor(y_train, dtype=torch.long), os.path.join(out_dir, "myeloma_train_labels.pt"))
    torch.save(torch.tensor(X_test, dtype=torch.float32), os.path.join(out_dir, "myeloma_test_data.pt"))
    torch.save(torch.tensor(y_test, dtype=torch.long), os.path.join(out_dir, "myeloma_test_labels.pt"))
    torch.save({"mean": torch.tensor(mean), "std": torch.tensor(std)}, os.path.join(out_dir, "myeloma_norm_stats.pt"))
    with open(os.path.join(out_dir, "myeloma_train_names.txt"), "w") as f:
        for n in names_train:
            f.write(n + "\n")
    with open(os.path.join(out_dir, "myeloma_test_names.txt"), "w") as f:
        for n in names_test:
            f.write(n + "\n")

    # PCA: fit on train, apply to test.
    n_components = max(1, min(100, len(X_train) - 1))
    print(f"\nFitting PCA (n_components={n_components}) on training data...")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA train shape: {X_train_pca.shape}")
    print(f"Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    torch.save(torch.tensor(X_train_pca, dtype=torch.float32), os.path.join(out_dir, "myeloma_train_pca.pt"))
    torch.save(torch.tensor(X_test_pca, dtype=torch.float32), os.path.join(out_dir, "myeloma_test_pca.pt"))
    torch.save(
        {
            "components": pca.components_,
            "mean": pca.mean_,
            "n_components": pca.n_components_,
        },
        os.path.join(out_dir, "myeloma_pca_model.pt"),
    )

    print(f"\nSaved outputs to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess multiple myeloma Hi-C tar archives with PCA.")
    parser.add_argument("--hic_dir", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/data")
    parser.add_argument("--out_dir", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed")
    parser.add_argument(
        "--label_mode",
        default="cell_line",
        choices=["cell_line", "all_tumor"],
        help="cell_line: RPMI8226=0, U266=1; all_tumor: all samples labeled 1",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    preprocess(
        hic_dir=args.hic_dir,
        out_dir=args.out_dir,
        label_mode=args.label_mode,
        test_size=args.test_size,
        random_state=args.random_state,
    )
