#!/usr/bin/env python3
"""
Training entry point for FIB-SEM 3D U-Net segmentation.

Usage:
    python train.py
"""

import os
import warnings
import zarr
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

from config import (
    GROUNDTRUTH_ROOT, RAW_ZARR_PATH, MODEL_DIR, LOG_DIR,
    CROP_IDS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    NUM_WORKERS, LOG_INTERVAL, REF_CLASS,
    SELECT_CLASSES, CLASS_ID_MAP, CLASS_WEIGHTS,
    MODEL_CONFIG, PATCH_Z, PATCH_Y, PATCH_X,
    PATCHES_PER_CROP, LOSS_CONFIG,
)
from src import UNet3D, CombinedDiceCELoss, load_one_crop, PatchSampler3D


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, log_interval):
    model.train()
    total_loss = 0.0

    for batch_idx, (raw, labels) in enumerate(dataloader):
        raw, labels = raw.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(raw)
        loss  = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"  [Epoch {epoch}, Batch {batch_idx + 1}] Loss: {avg:.4f}")

    return total_loss / len(dataloader)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("3D U-Net — Patch-Based Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ---- raw zarr ----
    print("Loading raw zarr (s1 scale)...")
    raw_zarr_dict = {"s1": zarr.open(os.path.join(RAW_ZARR_PATH, "s1"), mode="r")}
    print(f"  s1: {raw_zarr_dict['s1'].shape}\n")

    # ---- crops ----
    print("Loading crops...")
    crop_list = []
    for crop_id in CROP_IDS:
        try:
            crop_list.append(load_one_crop(
                crop_id, raw_zarr_dict, GROUNDTRUTH_ROOT,
                REF_CLASS, SELECT_CLASSES, CLASS_ID_MAP,
                RAW_ZARR_PATH, scale_name="s1"
            ))
        except Exception as e:
            print(f"  SKIP {crop_id}: {e}\n")

    if not crop_list:
        raise RuntimeError("No crops loaded — check paths in config.py")

    num_classes = len(SELECT_CLASSES) + 1   # +1 for background
    print(f"\nClasses: {num_classes}\n")

    # ---- dataset ----
    dataset = PatchSampler3D(
        crop_list=crop_list,
        patch_z=PATCH_Z, patch_y=PATCH_Y, patch_x=PATCH_X,
        patches_per_crop=PATCHES_PER_CROP,
        augment=True,
        num_classes=num_classes,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS)

    # ---- model ----
    print("Initialising 3D U-Net...")
    model = UNet3D(
        in_ch=MODEL_CONFIG['in_ch'],
        out_ch=MODEL_CONFIG['out_ch'],
        base_features=MODEL_CONFIG['base_features'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")

    # ---- loss / optimiser ----
    print("Setting up loss...")
    criterion = CombinedDiceCELoss(
        num_classes=num_classes,
        class_weights=CLASS_WEIGHTS,
        dice_weight=LOSS_CONFIG['dice_weight'],
        ce_weight=LOSS_CONFIG['ce_weight'],
        smooth=LOSS_CONFIG['smooth'],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"\n{'='*70}")
    print(f"Patch: {PATCH_Z}×{PATCH_Y}×{PATCH_X}  |  "
          f"Batch: {BATCH_SIZE}  |  Epochs: {NUM_EPOCHS}  |  "
          f"Patches/epoch: {len(dataset)}")
    print(f"{'='*70}\n")

    # ---- training loop ----
    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        loss = train_epoch(model, loader, criterion, optimizer,
                           device, epoch, LOG_INTERVAL)
        scheduler.step()
        print(f"  Avg loss: {loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}\n")

        # Checkpoint every 20 epochs
        if epoch % 20 == 0:
            ckpt = os.path.join(MODEL_DIR, f"unet3d_epoch{epoch:03d}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, ckpt)
            print(f"  ✓ Checkpoint: {ckpt}\n")

        # Best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet3d_best.pth"))

    # Final save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet3d_final.pth"))

    print(f"{'='*70}")
    print(f"Training complete — best loss: {best_loss:.4f}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
