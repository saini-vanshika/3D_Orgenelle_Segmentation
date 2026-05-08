#!/usr/bin/env python3
"""
Inference entry point for FIB-SEM 3D U-Net segmentation.

Usage:
    python inference.py                        # uses defaults in config
    python inference.py --crop crop234
    python inference.py --crop crop234 --model models/unet3d_best.pth
"""

import os
import argparse
import warnings
import zarr
import torch
import numpy as np

warnings.filterwarnings('ignore')

from config import (
    GROUNDTRUTH_ROOT, RAW_ZARR_PATH, MODEL_DIR, LOG_DIR,
    REF_CLASS, SELECT_CLASSES, CLASS_ID_MAP, MODEL_CONFIG,
    PATCH_Z, PATCH_Y, PATCH_X,
)
from src import (
    UNet3D,
    load_one_crop,
    predict_volume,
    calculate_dice_per_class, calculate_iou_per_class, print_metrics_table,
    visualize_slices,
)

CLASS_NAMES = ['background'] + list(SELECT_CLASSES.keys())


def parse_args():
    p = argparse.ArgumentParser(description="Run inference on one FIB-SEM crop")
    p.add_argument('--crop',  default='crop234',
                   help='Crop ID to run inference on (default: crop234)')
    p.add_argument('--model', default=None,
                   help='Path to .pth checkpoint (default: MODEL_DIR/unet3d_best.pth)')
    return p.parse_args()


def main():
    args       = parse_args()
    crop_id    = args.crop
    model_path = args.model or os.path.join(MODEL_DIR, "unet3d_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"INFERENCE — {crop_id}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Model:  {model_path}\n")

    # ---- raw zarr ----
    raw_zarr_dict = {"s1": zarr.open(os.path.join(RAW_ZARR_PATH, "s1"), mode='r')}
    print(f"Raw volume s1: {raw_zarr_dict['s1'].shape}\n")

    # ---- load crop ----
    crop_data = load_one_crop(
        crop_id, raw_zarr_dict, GROUNDTRUTH_ROOT,
        REF_CLASS, SELECT_CLASSES, CLASS_ID_MAP,
        RAW_ZARR_PATH, scale_name="s1"
    )
    raw_volume  = crop_data['raw']
    groundtruth = crop_data['label']
    print(f"Raw:  {raw_volume.shape} | GT labels: {np.unique(groundtruth)}\n")

    # ---- load model ----
    if not os.path.exists(model_path):
        print(f"ERROR: model not found at {model_path}")
        available = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')] \
                    if os.path.isdir(MODEL_DIR) else []
        if available:
            print("Available checkpoints:")
            for f in available:
                print(f"  {f}")
        return

    num_classes = len(SELECT_CLASSES) + 1
    model = UNet3D(
        in_ch=MODEL_CONFIG['in_ch'],
        out_ch=MODEL_CONFIG['out_ch'],
        base_features=MODEL_CONFIG['base_features'],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"✓ Model loaded — {num_classes} classes\n")

    # ---- inference ----
    print("Running sliding-window inference...")
    prediction = predict_volume(
        model, raw_volume, device,
        patch_z=PATCH_Z, patch_y=PATCH_Y, patch_x=PATCH_X,
        overlap=0.5,
    )
    print(f"Predicted labels: {np.unique(prediction)}\n")

    # ---- visualize ----
    Dz = raw_volume.shape[0]
    vis_path = os.path.join(LOG_DIR, f"{crop_id}_segmentation.png")
    visualize_slices(
        raw_volume, groundtruth, prediction,
        class_names=CLASS_NAMES,
        z_slices=[Dz // 4, Dz // 2, 3 * Dz // 4],
        output_path=vis_path,
    )

    # ---- metrics ----
    print(f"{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}")
    dice_scores = calculate_dice_per_class(prediction, groundtruth, num_classes)
    iou_scores  = calculate_iou_per_class(prediction, groundtruth, num_classes)
    mean_dice, mean_dice_nobg = print_metrics_table(dice_scores, iou_scores, CLASS_NAMES)

    # ---- save metrics ----
    metrics_path = os.path.join(LOG_DIR, f"{crop_id}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation — {crop_id}\n")
        f.write(f"Model: {model_path}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'ID':<8} {'Class':<20} {'Dice':<10} {'IoU':<10}\n")
        f.write("-" * 60 + "\n")
        for cid in sorted(dice_scores):
            name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"
            d, iou = dice_scores[cid], iou_scores[cid]
            if not np.isnan(d):
                f.write(f"{cid:<8} {name:<20} {d:<10.4f} {iou:<10.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean Dice (all):     {mean_dice:.4f}\n")
        f.write(f"Mean Dice (no bg):   {mean_dice_nobg:.4f}\n")

    print(f"✓ Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
