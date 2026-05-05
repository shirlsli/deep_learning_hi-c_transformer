#!/usr/bin/env python3
"""
Create a chr1 Hi-C heatmap from the first multiple myeloma tar.gz sample.
Saves output PNG in the multiple_myeloma folder.
"""

import argparse
import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np


def load_chr1_matrix_from_tar(tar_path: str, member_name: str) -> np.ndarray:
    with tarfile.open(tar_path, "r:gz") as tar_obj:
        member = tar_obj.getmember(member_name)
        fobj = tar_obj.extractfile(member)
        if fobj is None:
            raise ValueError(f"Could not read {member_name} from {tar_path}")

        rows = []
        for raw in fobj:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=" ", dtype=np.float32)
            if vals.size == 0:
                continue
            rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric rows found in {member_name}")

    matrix = np.vstack(rows)
    return matrix


def find_first_sample(data_dir: str) -> str:
    tar_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".tar.gz")])
    if not tar_files:
        raise ValueError(f"No .tar.gz files found in {data_dir}")
    return os.path.join(data_dir, tar_files[0])


def main():
    parser = argparse.ArgumentParser(description="Make chr1 heatmap from first multiple myeloma sample")
    parser.add_argument(
        "--data_dir",
        default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/data",
        help="Directory containing myeloma tar.gz files",
    )
    parser.add_argument(
        "--chr1_member",
        default="resolution_40k/cis/ice_normalization/chr1_40k_normalized_matrix.txt",
        help="Path inside tar for chr1 matrix",
    )
    parser.add_argument(
        "--out_png",
        default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/chr1_heatmap_first_sample.png",
        help="Output heatmap PNG path",
    )
    parser.add_argument(
        "--max_bins",
        type=int,
        default=300,
        help="Optional top-left crop size for readability; 0 means full matrix",
    )
    args = parser.parse_args()

    sample_tar = find_first_sample(args.data_dir)
    sample_name = os.path.basename(sample_tar)
    print(f"Using sample: {sample_name}")

    mat = load_chr1_matrix_from_tar(sample_tar, args.chr1_member)
    print(f"Loaded chr1 matrix shape: {mat.shape}")

    if args.max_bins > 0:
        n = min(args.max_bins, mat.shape[0], mat.shape[1])
        mat_plot = mat[:n, :n]
        print(f"Plotting top-left {n}x{n} region")
    else:
        mat_plot = mat
        print("Plotting full matrix")

    # Stabilize color scale for visualization.
    vmin = np.percentile(mat_plot, 1)
    vmax = np.percentile(mat_plot, 99)

    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat_plot, cmap="magma", origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="ICE-normalized contact")
    plt.title(f"Multiple Myeloma Hi-C chr1 Heatmap\n{sample_name}")
    plt.xlabel("chr1 bin index (40 kb)")
    plt.ylabel("chr1 bin index (40 kb)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=220)
    plt.close()

    print(f"Saved heatmap: {args.out_png}")


if __name__ == "__main__":
    main()
