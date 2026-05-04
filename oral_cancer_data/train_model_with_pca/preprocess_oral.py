#!/usr/bin/env python3
"""
Train a Transformer on oral cancer Hi-C data from multiple gzipped .matrix.gz files.
Uses the user's existing extract_chr1_profile and get_label functions unchanged.
Keeps log1p transform inside extract_chr1_profile and removes z-score normalization.
"""

import os
import glob
import gzip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.stats import pearsonr

HIC_DIR = "./HiC"
OUT_DIR = "./processed_w_z_norm"
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
CHR1_BINS = 6232
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

file_path = os.path.join(OUT_DIR, "oral_train_data.pt")
if os.path.exists(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File '{file_path}' does not exist.")

file_path = os.path.join(OUT_DIR, "oral_train_labels.pt")
if os.path.exists(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File '{file_path}' does not exist.")

file_path = os.path.join(OUT_DIR, "oral_data.pt")
if os.path.exists(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File '{file_path}' does not exist.")

file_path = os.path.join(OUT_DIR, "oral_labels.pt")
if os.path.exists(file_path):
    try:
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted successfully.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File '{file_path}' does not exist.")
    

def extract_chr1_profile(filepath):
    """Read sparse Hi-C matrix and return chr1 row-sum contact profile (6232-dim)."""
    profile = np.zeros(CHR1_BINS, dtype=np.float64)
    with gzip.open(filepath, "rt") as f:
        for line in f:
            parts = line.strip().split("	")
            b1, b2, val = int(parts[0]), int(parts[1]), float(parts[2])
            if b1 < CHR1_BINS and b2 < CHR1_BINS:
                profile[b1] += val
                if b1 != b2:
                    profile[b2] += val
    profile = np.log1p(profile)
    return profile


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
    print(f"Scanning Hi-C files in {hic_dir} ...", flush=True)
    hic_files = sorted(glob.glob(os.path.join(hic_dir, "*_40000_iced.matrix.gz")))
    print(f"Found {len(hic_files)} Hi-C files", flush=True)

    features = []
    labels = []
    names = []

    for fpath in hic_files:
        fname = os.path.basename(fpath)
        label = get_label(fname)
        print(f"Processing {fname} (label={'tumor' if label else 'normal'}) ...", flush=True)
        profile = extract_chr1_profile(fpath)
        features.append(profile)
        labels.append(label)
        names.append(fname)

    if len(features) == 0:
        raise ValueError(f"No files matched pattern in {hic_dir}")

    features = np.stack(features, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)

    # z-score normalize across samples (per feature/bin)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    features = (features - mean) / std

    torch.save(torch.tensor(features, dtype=torch.float32), os.path.join(out_dir, "oral_data.pt"))
    torch.save(torch.tensor(labels, dtype=torch.int64), os.path.join(out_dir, "oral_labels.pt"))
    print(f"Saved to {out_dir}/", flush=True)

    torch_data = torch.tensor(features, dtype=torch.float32)
    torch_labels = torch.tensor(labels, dtype=torch.float32)
    print(f"Data shape: {torch_data.shape} | dtype: {torch_data.dtype}", flush=True)
    print(f"Labels shape: {torch_labels.shape} | dtype: {torch_labels.dtype}", flush=True)

if __name__ == "__main__":
    preprocess(HIC_DIR, OUT_DIR)