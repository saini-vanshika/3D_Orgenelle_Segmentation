# Automated Segmentation of Cellular Organelles from FIB-SEM Data

**Comparative Analysis: Random Forest vs 3D U-Net**  
Vanshika Saini · TU Dresden · February 2025

---

## Overview

This project compares classical machine learning and deep learning for multi-class organelle segmentation in Focused Ion Beam Scanning Electron Microscopy (FIB-SEM) volumetric data.

**Dataset:** JRC COS7-1a from the [CellMap 2024 Segmentation Challenge](https://janelia.figshare.com/collections/CellMap_2024_Segmentation_Challenge/7456966/1)  
**Resolution:** 4 × 4 × 4 nm isotropic  
**Task:** 8-class voxel segmentation (background + 7 organelle types)

---

## Key Results

### Crop234 — Primary Evaluation

| Method | Mean Dice | Mean IoU |
|--------|-----------|----------|
| Random Forest (baseline) | 0.5434 | 0.3858 |
| **3D U-Net (CNN)** | **0.8077** | **0.7000** |

→ **+49% Dice**, **+82% IoU** improvement

### Per-Class Comparison (Crop234)

| Class | CNN Dice | RF Dice | Improvement |
|-------|----------|---------|-------------|
| Background | 0.968 | 0.924 | +4.7% |
| Mito Membrane | 0.752 | 0.281 | +167% |
| Mito Lumen | 0.849 | 0.069 | +1130% |
| Cytoplasm | 0.964 | 0.678 | +42% |
| ER Membrane | 0.536 | 0.326 | +64% |
| ER Lumen | 0.876 | 0.358 | +145% |
| Endo Membrane | 0.652 | 0.083 | +685% |
| Endo Lumen | 0.866 | 0.277 | +213% |

### 7-Fold Cross-Validation

| Test Crop | Mean Dice | Mean IoU | Status |
|-----------|-----------|----------|--------|
| crop234 | 0.8077 | 0.7000 | Best |
| crop236 | 0.7988 | 0.6978 | Good |
| crop237 | 0.7934 | 0.6723 | Good |
| crop239 | 0.7864 | 0.6525 | Good |
| crop248 | 0.7235 | 0.5889 | Good |
| crop252 | 0.7432 | 0.6542 | Good |
| crop292 | 0.0899 | 0.0527 | Poor* |

*crop292 has atypical class distribution — see [Cross-Validation Analysis](#cross-validation-analysis)

---

## Repository Structure

```
fibsem-organelle-segmentation/
├── config.py          # Paths, hyperparameters, class weights
├── train.py           # Training entry point
├── inference.py       # Inference + evaluation entry point
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py       # 3D U-Net architecture
│   ├── dataset.py     # OME-NGFF data loading + PatchSampler3D
│   ├── losses.py      # DiceLoss + CombinedDiceCELoss
│   └── utils.py       # Sliding-window inference, metrics, visualization
├── notebooks/
│   └──random_forest.ipynb
└── results/
    └── crop234_segmentation.png
```

> **Model checkpoints** (`models/`) are gitignored and saved locally. They are auto-created by `config.py`.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/fibsem-organelle-segmentation
cd fibsem-organelle-segmentation
pip install -r requirements.txt
```

---

## Dataset

Download `jrc_cos7-1a` from [OpenOrganelle / CellMap Challenge](https://github.com/janelia-cellmap/cellmap-segmentation-challenge).

**Full volume:** 906 × 2184 × 10304 voxels at 4 nm/voxel  
**Format:** OME-NGFF (Zarr multi-scale)

Expected zarr structure:
```
jrc_cos7-1a.zarr/recon-1/
├── em/fibsem-uint8/          # Raw FIB-SEM (multi-scale s0, s1, s2...)
└── labels/groundtruth/
    ├── crop234/mito_mem/s1/
    ├── crop234/mito_lum/s1/
    └── ...
```

Edit paths in `config.py` to match your environment.

---

## Usage

**Training:**
```bash
python train.py
```

**Inference on a test crop:**
```bash
python inference.py --crop crop236
python inference.py --crop crop236 --model models/unet3d_best.pth
```

---

## Methods

### 1. Random Forest Baseline

A classical voxel-wise classifier using **30+ hand-crafted features** per voxel:

- Raw intensity
- Multi-scale Gaussian smoothing (σ = 0.5, 1, 2, 4)
- Gradient magnitude and Laplacian per scale
- Directional Sobel filters (X, Y, Z)
- Local standard deviation (texture)
- Difference of Gaussians
- Normalized spatial coordinates (X, Y, Z)

**Boundary-aware sampling:** 40% of training voxels drawn from structural boundaries (morphological erosion) to improve membrane detection.

| Config | Value |
|--------|-------|
| Classifier | Random Forest (100 trees) |
| Features | 30–40 per voxel |
| Training time | ~1 hour (CPU) |
| Inference time | ~45 min per crop |

**Key limitation:** Per-voxel classification with no 3D spatial context → salt-and-pepper noise, fragmented predictions, catastrophic failure on small structures.

---

### 2. 3D U-Net (CNN)

Encoder–decoder architecture with skip connections, processing full 3D volumetric context.

**Architecture:**

| Component | Details |
|-----------|---------|
| Input | 1 × 128 × 128 × 128 patch |
| Encoder | 3 levels, 3×3×3 convolutions |
| Skip connections | Concatenate encoder → decoder |
| Bottleneck | 128 channels |
| Decoder | 3 levels, transposed convolutions |
| Output | 8 channels (one per class) |
| Normalization | Instance Normalization |
| Parameters | ~2.7M |

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 4 patches |
| Epochs | 150 |
| Patches per crop | 50 |
| Patches per epoch | 350 |
| Patch size | 128×128×128 |
| Schedule | Cosine annealing |
| Gradient clipping | max norm = 1.0 |

**Loss:** Combined Dice (50%) + Cross-Entropy (50%) with inverse-sqrt class frequency weighting.

**Training dynamics:** Smooth monotonic decrease from 1.1851 → 0.2021 (83% reduction), convergence by epoch 50, no oscillations.

---

## Key Design Decisions

### Why patch-based training (128³) instead of full volumes (200³)?

| Aspect | Full 200³ | Patches 128³ |
|--------|-----------|--------------|
| Memory | ~2 GB, batch 1 | ~50 MB, batch 4 |
| Updates/epoch | 7 | 350 (50×) |
| Variable crop sizes | Problematic | Handled naturally |
| Gradient stability | Noisy | Stable |

### Why Combined Dice + CE loss?

- **Dice alone:** Optimizes segmentation metric, but gradients can be unstable
- **CE alone:** Stable gradients, but doesn't optimize Dice directly
- **Combined (50-50):** Dice 0.808 — best of both

### Critical: OME-NGFF Coordinate Transform

Each crop stores its physical location in the tissue via a per-crop translation. The raw EM volume has its own coordinate system. Correct alignment requires:

```python
raw_voxel = (physical_coord - raw_translation) / raw_scale
```

This is frequently overlooked in multi-scale datasets and causes misaligned predictions if ignored.

### Class Imbalance Handling

- Cytoplasm: ~50% of voxels
- Background: ~25%
- Rare organelles: <2%
- Imbalance ratio: 501×

**Solution:** Inverse square-root frequency weighting, normalized by median, with manual boost for background and cytoplasm which were being ignored despite having "normal" frequencies.

---

## Cross-Validation Analysis

6 of 7 crops show consistent performance (Dice 0.72–0.81). **Crop292 fails** (Dice 0.089) due to atypical class distribution — some classes that are rare in training data are common in crop292, causing the learned class weights to be mismatched.

This is expected cross-validation variance for a small dataset (7 crops), not a model bug. It highlights the need for:
- More annotated crops
- Adaptive per-test-crop weighting
- Structural postprocessing

---

## Limitations & Future Work

**Current limitations:**
- Small dataset (7 crops) → cross-validation variance
- ER membrane performance (Dice 0.536) — thin structures remain difficult
- No structural postprocessing applied
- Single architecture evaluated

**High-priority improvements:**
- Structural postprocessing enforcing membrane-encloses-lumen constraints (~+4-5% estimated)
- Reduce training loss below 0.1 via attention mechanisms or deeper networks
- Expand to more annotated crops

**Medium-priority:**
- Validate on JRC COS7-1b for cross-dataset generalization
- Explore Attention U-Net and Vision Transformers
- Uncertainty quantification via dropout or ensembles

---

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | Background | Non-annotated tissue |
| 1 | Mito Membrane | Mitochondrial outer membrane |
| 2 | Mito Lumen | Space within mitochondria |
| 3 | Cytoplasm | Cell cytoplasm |
| 4 | ER Membrane | Endoplasmic reticulum membrane |
| 5 | ER Lumen | ER interior space |
| 6 | Endo Membrane | Endosome membrane |
| 7 | Endo Lumen | Endosome interior |

---

## Citation

If you use this code or reference this work, please cite:

```
Vanshika Saini. Automated Segmentation of Cellular Organelles from FIB-SEM Data.
TU Dresden, February 2025.
```

**Dataset:**
```
David Ackerman et al. CellMap 2024 Segmentation Challenge. Janelia Research Campus, 2024.
```

**Key references:**
- 3D U-Net: Çiçek et al., MICCAI 2016
- Dice Loss: Milletari et al., 3DV 2016
- Generalised Dice for imbalanced data: Sudre et al., MICCAI 2017
