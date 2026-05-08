"""
Inference utilities, evaluation metrics, and visualization.

Functions:
    predict_volume()        — sliding-window prediction over a full volume
    calculate_dice_per_class() — per-class Dice score
    calculate_iou_per_class()  — per-class IoU score
    print_metrics_table()   — formatted console + return mean scores
    create_colormap()       — fixed color map for label overlay
    visualize_slices()      — side-by-side raw / GT / prediction figure
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Sliding-window inference
# ---------------------------------------------------------------------------

def predict_volume(model, raw_volume: np.ndarray, device,
                   patch_z: int = 128, patch_y: int = 128, patch_x: int = 128,
                   overlap: float = 0.5) -> np.ndarray:
    """
    Predict segmentation for an entire volume using a sliding window.

    Overlapping patches are averaged (soft voting) before argmax, which
    reduces boundary artefacts compared to hard tiling.

    Args:
        model:              Trained UNet3D (eval mode set internally)
        raw_volume:         uint8 ndarray [Z, Y, X]
        device:             torch.device
        patch_z/y/x:        Patch size for each axis
        overlap:            Fraction of patch that overlaps neighbours (0–1)

    Returns:
        prediction:         uint8 ndarray [Z, Y, X] with class indices
    """
    model.eval()

    Dz, Dy, Dx    = raw_volume.shape
    num_classes   = model.outc.out_channels
    stride_z      = max(1, int(patch_z * (1 - overlap)))
    stride_y      = max(1, int(patch_y * (1 - overlap)))
    stride_x      = max(1, int(patch_x * (1 - overlap)))

    prediction_sum = np.zeros((num_classes, Dz, Dy, Dx), dtype=np.float32)
    count          = np.zeros((Dz, Dy, Dx), dtype=np.float32)

    def _positions(dim_size, patch_size, stride):
        """Generate start positions that guarantee full coverage."""
        if dim_size <= patch_size:
            return [0]
        pos = list(range(0, dim_size - patch_size + 1, stride))
        if not pos or pos[-1] + patch_size < dim_size:
            pos.append(dim_size - patch_size)
        return pos

    z_pos = _positions(Dz, patch_z, stride_z)
    y_pos = _positions(Dy, patch_y, stride_y)
    x_pos = _positions(Dx, patch_x, stride_x)

    total = len(z_pos) * len(y_pos) * len(x_pos)
    print(f"  Volume: {Dz}×{Dy}×{Dx} | "
          f"Patch: {patch_z}×{patch_y}×{patch_x} | "
          f"Stride: {stride_z}×{stride_y}×{stride_x} | "
          f"Patches: {total}")

    done = 0
    with torch.no_grad():
        for z0 in z_pos:
            for y0 in y_pos:
                for x0 in x_pos:
                    z1, y1, x1 = z0 + patch_z, y0 + patch_y, x0 + patch_x

                    patch = raw_volume[z0:z1, y0:y1, x0:x1]

                    # Zero-pad if the crop is smaller than the patch
                    if patch.shape != (patch_z, patch_y, patch_x):
                        padded = np.zeros((patch_z, patch_y, patch_x), dtype=patch.dtype)
                        padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                        patch = padded

                    t = torch.from_numpy(patch.astype(np.float32) / 255.0)
                    t = t.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,Z,Y,X)

                    probs = F.softmax(model(t), dim=1).cpu().numpy()[0]  # (C,Z,Y,X)

                    # Actual extent (may be smaller than patch at volume edge)
                    az, ay, ax = min(z1, Dz) - z0, min(y1, Dy) - y0, min(x1, Dx) - x0
                    prediction_sum[:, z0:z0+az, y0:y0+ay, x0:x0+ax] += probs[:, :az, :ay, :ax]
                    count[z0:z0+az, y0:y0+ay, x0:x0+ax] += 1

                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"    {done}/{total} patches", end='\r')

    print(f"\n  ✓ Done — {total} patches processed")

    uncovered = int(np.sum(count == 0))
    if uncovered:
        print(f"  ⚠ WARNING: {uncovered} uncovered voxels")

    averaged    = prediction_sum / (count[np.newaxis] + 1e-8)
    return np.argmax(averaged, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def calculate_dice_per_class(prediction: np.ndarray,
                              groundtruth: np.ndarray,
                              num_classes: int) -> dict:
    """
    Compute per-class Dice score.

    Returns:
        {class_id: float | nan}   nan when class absent in ground truth
    """
    pred = prediction.flatten()
    gt   = groundtruth.flatten()
    scores = {}

    for c in range(num_classes):
        p_mask = pred == c
        g_mask = gt   == c
        if g_mask.sum() == 0:
            scores[c] = float('nan')
        else:
            scores[c] = 2.0 * np.sum(p_mask & g_mask) / (p_mask.sum() + g_mask.sum() + 1e-8)

    return scores


def calculate_iou_per_class(prediction: np.ndarray,
                             groundtruth: np.ndarray,
                             num_classes: int) -> dict:
    """
    Compute per-class IoU (Jaccard index).

    Returns:
        {class_id: float | nan}
    """
    pred = prediction.flatten()
    gt   = groundtruth.flatten()
    scores = {}

    for c in range(num_classes):
        p_mask = pred == c
        g_mask = gt   == c
        union  = np.sum(p_mask | g_mask)
        scores[c] = float('nan') if union == 0 else np.sum(p_mask & g_mask) / union

    return scores


def print_metrics_table(dice_scores: dict, iou_scores: dict,
                         class_names: list) -> tuple:
    """
    Print a formatted per-class metrics table and return mean scores.

    Returns:
        (mean_dice_all, mean_dice_no_bg)
    """
    W = 75
    print("\n" + "=" * W)
    print("PER-CLASS EVALUATION METRICS")
    print("=" * W)
    print(f"{'ID':<5} {'Class':<20} {'Dice':>10} {'IoU':>10}")
    print("-" * W)

    for cid in sorted(dice_scores):
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        d    = dice_scores[cid]
        iou  = iou_scores[cid]
        if np.isnan(d):
            print(f"{cid:<5} {name:<20} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"{cid:<5} {name:<20} {d:>10.4f} {iou:>10.4f}")

    valid_d   = [v for k, v in dice_scores.items() if not np.isnan(v)]
    valid_d_nobg = [v for k, v in dice_scores.items() if not np.isnan(v) and k > 0]
    mean_all  = float(np.mean(valid_d))       if valid_d      else 0.0
    mean_nobg = float(np.mean(valid_d_nobg))  if valid_d_nobg else 0.0

    print("-" * W)
    print(f"{'':5} {'Mean (all classes)':<20} {mean_all:>10.4f}")
    print(f"{'':5} {'Mean (no background)':<20} {mean_nobg:>10.4f}")
    print("=" * W + "\n")

    return mean_all, mean_nobg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Fixed per-class colours (RGB, 0-1)
_CLASS_COLORS = [
    [0.00, 0.00, 0.00],   # 0  background  — black
    [1.00, 0.00, 0.00],   # 1  mito_mem    — red
    [0.00, 1.00, 0.00],   # 2  mito_lum    — green
    [0.00, 0.00, 1.00],   # 3  cyto        — blue
    [1.00, 1.00, 0.00],   # 4  endo_mem    — yellow
    [1.00, 0.00, 1.00],   # 5  endo_lum    — magenta
    [0.00, 1.00, 1.00],   # 6  mt_in       — cyan
    [0.50, 0.50, 0.00],   # 7  mt_out      — olive
    [0.50, 0.00, 0.50],   # 8  er_mem      — purple
    [0.00, 0.50, 0.50],   # 9  er_lum      — teal
    [1.00, 0.50, 0.00],   # 10 ves_mem     — orange
    [0.50, 1.00, 0.00],   # 11 ves_lum     — lime
]


def create_colormap(num_classes: int) -> ListedColormap:
    """Return a ListedColormap with a fixed colour per organelle class."""
    colors = _CLASS_COLORS[:num_classes]
    while len(colors) < num_classes:
        colors.append(list(np.random.rand(3)))
    return ListedColormap(colors)


def visualize_slices(raw: np.ndarray,
                     groundtruth: np.ndarray,
                     prediction: np.ndarray,
                     class_names: list,
                     z_slices: list = None,
                     output_path: str = None):
    """
    Save / display a figure with N rows (one per z-slice), 3 columns:
    Raw EM  |  Ground Truth  |  Prediction

    Args:
        raw:          uint8 [Z, Y, X]
        groundtruth:  uint8 [Z, Y, X]
        prediction:   uint8 [Z, Y, X]
        class_names:  list of names indexed by class id
        z_slices:     z indices to display (default: 25%, 50%, 75%)
        output_path:  if given, save figure to this path
    """
    Dz = raw.shape[0]
    if z_slices is None:
        z_slices = [Dz // 4, Dz // 2, 3 * Dz // 4]

    num_classes = int(max(groundtruth.max(), prediction.max())) + 1
    cmap        = create_colormap(num_classes)
    n           = len(z_slices)

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, z in enumerate(z_slices):
        axes[i, 0].imshow(raw[z], cmap='gray')
        axes[i, 0].set_title(f'Raw EM  (z={z})')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(groundtruth[z], cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[i, 1].set_title(f'Ground Truth  (z={z})')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(prediction[z], cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[i, 2].set_title(f'Prediction  (z={z})')
        axes[i, 2].axis('off')

    # Legend
    legend_elems = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(c), markersize=10,
                   label=f'{c}: {class_names[c] if c < len(class_names) else c}')
        for c in range(num_classes)
    ]
    fig.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure: {output_path}")

    plt.show()
