#!/usr/bin/env python3
"""
Fine-tune on ALL multiple myeloma samples (combine train + test).
No holdout set — just fine-tune on all 4 samples.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


ORAL_INPUT_DIM = 76
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
DEFAULT_ORAL_PCA_PATH = "/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/processed_20260329_133858/pca_model.pt"
DEFAULT_ORAL_CHECKPOINT_PATH = "/athena/cayuga_0019/scratch/ssl4003/deep-learning/oral_cancer_data/train_model_with_pca/checkpoints_20260329_133858/best_model.pt"


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


def load_pretrained_for_classification(pretrained_path, device):
    checkpoint = torch.load(pretrained_path, map_location=device)
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    model = HiCTransformer(
        input_dim=ORAL_INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=2,
    )

    # Load all weights except classification head
    incompatible_keys = []
    for k in list(state_dict.keys()):
        if k.startswith("head."):
            incompatible_keys.append(k)

    for k in incompatible_keys:
        state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    print(f"Transferred {len(state_dict)} tensors | New (random init): {incompatible_keys}")
    return model.to(device)


def apply_freeze(model, freeze_option):
    if freeze_option == 0:
        # Freeze encoder + input_proj, train norm + head
        for name, param in model.named_parameters():
            if name.startswith("input_proj") or name.startswith("encoder"):
                param.requires_grad = False
    elif freeze_option == 1:
        # Freeze all except head
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model


def project_with_oral_pca(data_tensor, pca_model_path):
    pca_obj = torch.load(pca_model_path, map_location="cpu", weights_only=False)
    components = torch.tensor(np.asarray(pca_obj["components"], dtype=np.float32))
    mean = torch.tensor(np.asarray(pca_obj["mean"], dtype=np.float32))

    device = data_tensor.device
    components = components.to(device)
    mean = mean.to(device)

    if data_tensor.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Feature dim {data_tensor.shape[1]} does not match oral PCA input dim {mean.shape[0]}"
        )

    centered = data_tensor - mean
    projected = centered @ components.T
    return projected.float()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on ALL myeloma samples (no train/test split)")
    parser.add_argument("--train_data", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_train_data.pt")
    parser.add_argument("--train_labels", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_train_labels.pt")
    parser.add_argument("--test_data", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_data.pt")
    parser.add_argument("--test_labels", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/processed/myeloma_test_labels.pt")
    parser.add_argument("--pretrained_path", default=DEFAULT_ORAL_CHECKPOINT_PATH)
    parser.add_argument("--pca_model_path", default=DEFAULT_ORAL_PCA_PATH)
    parser.add_argument("--output_dir", default="/athena/angsd/scratch/ssl4003/deep-learning/multiple_myeloma/checkpoints_all_samples")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--freeze_option", type=int, default=None, help="0=freeze encoder, 1=freeze all but head, None=both")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all data (train + test combined)
    train_data = torch.load(args.train_data, map_location="cpu").float()
    train_labels = torch.load(args.train_labels, map_location="cpu").long()
    test_data = torch.load(args.test_data, map_location="cpu").float()
    test_labels = torch.load(args.test_labels, map_location="cpu").long()

    # Combine
    all_data = torch.cat([train_data, test_data], dim=0)
    all_labels = torch.cat([train_labels, test_labels], dim=0)

    print(f"\nCombined dataset: {all_data.shape[0]} samples, {all_data.shape[1]} features")
    print(f"Label distribution: {all_labels.sum().item()} class-1, {len(all_labels) - all_labels.sum().item()} class-0")

    # Project if needed
    if all_data.shape[1] != ORAL_INPUT_DIM:
        print(f"Projecting with oral PCA: {args.pca_model_path}")
        all_data = project_with_oral_pca(all_data, args.pca_model_path)
        print(f"Projected to {all_data.shape[1]}D")

    dataloader = DataLoader(TensorDataset(all_data, all_labels), batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    lr_map = {0: 1e-6, 1: 1e-4}
    label_map = {
        0: "freeze encoder+input_proj, train norm+head",
        1: "freeze all except classification head",
    }

    freeze_opts = [args.freeze_option] if args.freeze_option is not None else [0, 1]
    os.makedirs(args.output_dir, exist_ok=True)

    for freeze_opt in freeze_opts:
        print("\n" + "=" * 70)
        print(f" Fine-tuning freeze_option={freeze_opt}: {label_map[freeze_opt]}")
        print(f" LR={lr_map[freeze_opt]:.0e} epochs={args.epochs} batch_size={args.batch_size}")
        print("=" * 70)

        model = apply_freeze(load_pretrained_for_classification(args.pretrained_path, device), freeze_opt)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_map[freeze_opt])

        best_loss = float("inf")
        best_state = None

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, dataloader, criterion, optimizer, device)
            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}/{args.epochs} Loss: {loss:.4f}")

        save_path = os.path.join(args.output_dir, f"myeloma_allsamples_freeze{freeze_opt}_best.pt")
        torch.save(best_state, save_path)
        print(f"  Best model saved -> {save_path} (loss={best_loss:.4f})")

    print("\nFine-tuning complete!")


if __name__ == "__main__":
    main()
