"""
Configuration for FIB-SEM 3D U-Net segmentation.
Edit paths below to match your environment.
"""

import os

# ----------------------------------------------------------------------------
# Paths  (edit these)
# ----------------------------------------------------------------------------
PROJECT_ROOT     = "/data/horse/ws/vasa754g-segmentation"
DATA_ROOT        = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR        = os.path.join(PROJECT_ROOT, "models")
LOG_DIR          = os.path.join(PROJECT_ROOT, "result")

GROUNDTRUTH_ROOT = os.path.join(DATA_ROOT, "jrc_cos7-1a.zarr/recon-1/labels/groundtruth")
RAW_ZARR_PATH    = os.path.join(DATA_ROOT, "jrc_cos7-1a.zarr/recon-1/em/fibsem-uint8")

# ----------------------------------------------------------------------------
# Crops
# ----------------------------------------------------------------------------
# crop236 is held out as the test set
CROP_IDS = ["crop236", "crop237", "crop239", "crop248", "crop252", "crop254", "crop292"]
REF_CLASS = "mito_mem"   # Reference class used to read crop coordinate metadata

# ----------------------------------------------------------------------------
# Classes  (background = 0 implicitly)
# ----------------------------------------------------------------------------
SELECT_CLASSES = {
    'mito_mem': 1,
    'mito_lum': 2,
    'cyto':     3,
    'endo_mem': 4,
    'endo_lum': 5,
    'mt_in':    6,
    'mt_out':   7,
    'er_mem':   8,
    'er_lum':   9,
}
CLASS_ID_MAP = SELECT_CLASSES.copy()

# Per-class loss weights (length must equal len(SELECT_CLASSES) + 1)
# Index 0 = background, indices 1-9 match SELECT_CLASSES values above.
# Background and cytoplasm (idx 3) get manual boosts to prevent the model
# from ignoring the two most common classes.
CLASS_WEIGHTS = [1.5, 2.0, 2.0, 1.2, 3.0, 3.0, 3.0, 3.0, 2.5, 2.5]

# ----------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------
BATCH_SIZE    = 4
NUM_EPOCHS    = 200
LEARNING_RATE = 1e-4
NUM_WORKERS   = 4
LOG_INTERVAL  = 10        # Print loss every N batches

# Patch sampling
PATCH_Z          = 128
PATCH_Y          = 128
PATCH_X          = 128
PATCHES_PER_CROP = 50     # Random patches drawn per crop per epoch

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
MODEL_CONFIG = {
    'in_ch':         1,
    'out_ch':        len(SELECT_CLASSES) + 1,   # 10
    'base_features': 32,
}

# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------
LOSS_CONFIG = {
    'dice_weight': 0.5,
    'ce_weight':   0.5,
    'smooth':      1e-5,
}

# ----------------------------------------------------------------------------
# Auto-create output directories
# ----------------------------------------------------------------------------
for _d in [MODEL_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)
