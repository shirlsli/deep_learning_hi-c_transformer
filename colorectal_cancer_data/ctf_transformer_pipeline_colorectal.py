"""
Fine-tuning pipeline for colorectal cancer Hi-C data.

Loads the pretrained HiCTransformer (trained on oral cancer via train_oral.py),
replaces its regression head with a 2-class classification head, and fine-tunes
on colorectal cancer data with two freeze strategies:

  freeze_option=0 : freeze encoder + input_proj  →  train norm + head  (lr=1e-6)
  freeze_option=1 : freeze all except head        →  train head only    (lr=1e-4)

The pretrained checkpoint is saved as a dict:
  {"epoch": ..., "model_state": ..., "optimizer_state": ..., "val_pearson": ..., "val_mse": ...}
Fine-tuned checkpoints are saved as flat state dicts for evaluate_colorectal.py.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Architecture constants — must match oral cancer best_model.pt ─────────────
ORAL_INPUT_DIM     = 76
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
            x = x.unsqueeze(1)           # (batch, 1, input_dim)
        x = self.input_proj(x)           # (batch, 1, d_model)
        x = self.encoder(x)              # (batch, 1, d_model)
        x = self.norm(x.mean(dim=1))     # mean-pool → (batch, d_model)
        return self.head(x)              # (batch, output_dim)


def load_pretrained_for_classification(pretrained_path, device, input_dim):
    """
    Load oral cancer checkpoint and transfer weights to a 2-class classification model.
    The pretrained head is a regression head (output_dim=1); it is replaced with a
    fresh classification head (output_dim=2) using strict=False loading.
    input_proj is excluded from transfer because input_dim may differ from ORAL_INPUT_DIM.
    """
    ckpt = torch.load(pretrained_path, map_location=device)
    epoch     = ckpt.get("epoch", "?")
    val_pcc   = ckpt.get("val_pearson", float("nan"))
    print(f"  Loaded checkpoint: epoch={epoch}, val_pearson={val_pcc:.4f}")
    print(f"  Building model with input_dim={input_dim} (pretrained used {ORAL_INPUT_DIM})")

    model = HiCTransformer(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        output_dim=2          # classification head
    ).to(device)

    # Transfer encoder + norm only; input_proj and head are randomly initialized
    pretrained_sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
    transfer_sd   = {k: v for k, v in pretrained_sd.items()
                     if not k.startswith("head") and not k.startswith("input_proj")}
    missing, _    = model.load_state_dict(transfer_sd, strict=False)
    print(f"  Transferred {len(transfer_sd)} tensors | New (random init): {missing}")
    return model


def apply_freeze(model, freeze_option):
    """
    freeze_option=0 : freeze encoder + input_proj, train norm + head
    freeze_option=1 : freeze all except classification head
    """
    for param in model.parameters():
        param.requires_grad = True

    if freeze_option == 0:
        for param in model.input_proj.parameters():
            param.requires_grad = False
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif freeze_option == 1:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            total_loss += criterion(model(X), y).item()
    return total_loss / len(dataloader)


def finetune_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load preprocessed tensors (labels must be long for CrossEntropyLoss)
    train_data   = torch.load(args.train_data,   map_location="cpu").float()
    train_labels = torch.load(args.train_labels, map_location="cpu").long()
    test_data    = torch.load(args.test_data,    map_location="cpu").float()
    test_labels  = torch.load(args.test_labels,  map_location="cpu").long()

    input_dim = train_data.shape[-1]
    print(f"Train: {len(train_labels)} samples  "
          f"({train_labels.sum().item()} tumor, {(train_labels==0).sum().item()} normal)")
    print(f"Test:  {len(test_labels)} samples  "
          f"({test_labels.sum().item()} tumor, {(test_labels==0).sum().item()} normal)")
    print(f"Input dim: {input_dim}")

    train_loader = DataLoader(TensorDataset(train_data, train_labels),
                              batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(test_data,  test_labels),
                              batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # Learning rates per freeze strategy
    lr_map    = {0: 1e-6, 1: 1e-4}
    label_map = {
        0: "freeze encoder+input_proj, train norm+head",
        1: "freeze all except classification head",
    }

    freeze_opts = [args.freeze_option] if args.freeze_option is not None else [0, 1]

    for freeze_opt in freeze_opts:
        print(f"\n{'='*65}")
        print(f" Fine-tuning  freeze_option={freeze_opt}: {label_map[freeze_opt]}")
        print(f" LR={lr_map[freeze_opt]:.0e}   epochs={args.epochs}")
        print(f"{'='*65}")

        model     = apply_freeze(
            load_pretrained_for_classification(args.pretrained_path, device, input_dim),
            freeze_opt
        )
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr_map[freeze_opt]
        )

        best_loss  = float("inf")
        best_state = None

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
            if loss < best_loss:
                best_loss  = loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{args.epochs}  Train Loss: {loss:.4f}")

        # Save fine-tuned model as flat state dict
        save_path = os.path.join(args.output_dir,
                                 f"colorectal_cancer_freeze{freeze_opt}_best.pt")
        torch.save(best_state, save_path)
        print(f"  Best model saved → {save_path}  (best loss={best_loss:.4f})")

        # Quick test loss on held-out set
        model.load_state_dict(best_state)
        model.to(device)
        test_loss = eval_epoch(model, test_loader, criterion, device)
        print(f"  Test CE Loss: {test_loss:.4f}")

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained HiCTransformer (oral cancer) on colorectal cancer Hi-C"
    )
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to oral cancer best_model.pt (dict checkpoint)")
    parser.add_argument("--train_data",      type=str, required=True,
                        help="Path to colorectal_train_data.pt")
    parser.add_argument("--train_labels",    type=str, required=True,
                        help="Path to colorectal_train_labels.pt")
    parser.add_argument("--test_data",       type=str, required=True,
                        help="Path to colorectal_test_data.pt")
    parser.add_argument("--test_labels",     type=str, required=True,
                        help="Path to colorectal_test_labels.pt")
    parser.add_argument("--epochs",          type=int, default=50,
                        help="Number of fine-tuning epochs (default: 50)")
    parser.add_argument("--freeze_option",   type=int, choices=[0, 1], default=None,
                        help="0=freeze encoder+input_proj; 1=freeze all except head; "
                             "omit to run both strategies sequentially")
    parser.add_argument("--output_dir",      type=str, default=".",
                        help="Directory to save fine-tuned checkpoints (default: cwd)")
    args = parser.parse_args()
    finetune_pipeline(args)
