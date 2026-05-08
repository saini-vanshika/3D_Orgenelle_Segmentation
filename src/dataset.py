"""
Data loading and patch sampling for FIB-SEM segmentation.

- load_one_crop():  Reads raw EM + ground-truth labels from OME-NGFF zarr.
                    Handles per-crop coordinate transforms (physical → voxel).
- PatchSampler3D:   PyTorch Dataset that draws random 3D patches + augmentation.
"""

import os
import json
from typing import Dict, List

import numpy as np
import zarr
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Crop loading
# ---------------------------------------------------------------------------

def load_one_crop(crop_id: str,
                  raw_zarr_dict: dict,
                  groundtruth_root: str,
                  ref_class: str,
                  select_classes: Dict,
                  class_id_map: Dict,
                  raw_zarr_path: str,
                  scale_name: str = "s1") -> dict:
    """
    Load one labelled crop from the OME-NGFF zarr store.

    CRITICAL: Each crop stores its physical location in the tissue via a
    per-crop translation.  The raw EM volume has its own translation.
    We must convert the crop's physical region to raw voxel coordinates.

    Formula:  raw_voxel = (physical_coord - raw_translation) / raw_scale

    Args:
        crop_id:          e.g. "crop236"
        raw_zarr_dict:    {scale_name: zarr.Array} loaded outside this function
        groundtruth_root: path to groundtruth root directory
        ref_class:        reference class used to read crop metadata (e.g. "mito_mem")
        select_classes:   {class_name: class_id} to load
        class_id_map:     same mapping (used for label assignment)
        raw_zarr_path:    path to raw EM zarr (needed to read .zattrs)
        scale_name:       which resolution level to use (default "s1")

    Returns:
        dict with keys: 'raw' (uint8 ndarray), 'label' (uint8 ndarray),
                        'shape' (tuple), 'id' (str)
    """
    crop_root     = os.path.join(groundtruth_root, crop_id)
    ref_class_path = os.path.join(crop_root, ref_class)
    ref_class_zattr = os.path.join(ref_class_path, ".zattrs")

    # ---- label metadata ----
    with open(ref_class_zattr) as f:
        label_attrs = json.load(f)

    datasets = label_attrs["multiscales"][0]["datasets"]
    scale_dataset = next((d for d in datasets if d.get("path") == scale_name), None)
    if scale_dataset is None:
        raise ValueError(f"Scale {scale_name} not found in label metadata for {crop_id}")

    label_scale       = [1.0, 1.0, 1.0]
    label_translation = [0.0, 0.0, 0.0]
    for t in scale_dataset.get("coordinateTransformations", []):
        if t["type"] == "scale":       label_scale       = t["scale"]
        elif t["type"] == "translation": label_translation = t["translation"]

    # ---- raw metadata ----
    with open(os.path.join(raw_zarr_path, ".zattrs")) as f:
        raw_attrs = json.load(f)

    raw_datasets = raw_attrs["multiscales"][0]["datasets"]
    raw_scale_dataset = next((d for d in raw_datasets if d.get("path") == scale_name), None)
    if raw_scale_dataset is None:
        raise ValueError(f"Scale {scale_name} not found in raw metadata")

    raw_scale       = [1.0, 1.0, 1.0]
    raw_translation = [0.0, 0.0, 0.0]
    for t in raw_scale_dataset.get("coordinateTransformations", []):
        if t["type"] == "scale":       raw_scale       = t["scale"]
        elif t["type"] == "translation": raw_translation = t["translation"]

    # ---- crop dimensions ----
    ref_arr = zarr.open(os.path.join(ref_class_path, scale_name), mode="r")
    Dz, Dy, Dx = ref_arr.shape

    # ---- physical → raw voxel coordinates ----
    lz0, ly0, lx0 = label_translation
    lz1 = lz0 + Dz * label_scale[0]
    ly1 = ly0 + Dy * label_scale[1]
    lx1 = lx0 + Dx * label_scale[2]

    raw_vz0 = int(round((lz0 - raw_translation[0]) / raw_scale[0]))
    raw_vy0 = int(round((ly0 - raw_translation[1]) / raw_scale[1]))
    raw_vx0 = int(round((lx0 - raw_translation[2]) / raw_scale[2]))
    raw_vz1 = int(round((lz1 - raw_translation[0]) / raw_scale[0]))
    raw_vy1 = int(round((ly1 - raw_translation[1]) / raw_scale[1]))
    raw_vx1 = int(round((lx1 - raw_translation[2]) / raw_scale[2]))

    # Force match to label size (rounding can cause off-by-one)
    if (raw_vz1 - raw_vz0, raw_vy1 - raw_vy0, raw_vx1 - raw_vx0) != (Dz, Dy, Dx):
        print(f"  WARNING: size rounding mismatch for {crop_id}, forcing match.")
        raw_vz1, raw_vy1, raw_vx1 = raw_vz0 + Dz, raw_vy0 + Dy, raw_vx0 + Dx

    print(f"{crop_id} ({scale_name}): shape={( Dz, Dy, Dx)}, "
          f"raw voxels z=[{raw_vz0}:{raw_vz1}]")

    # ---- bounds check ----
    raw_zarr = raw_zarr_dict[scale_name]
    raw_shape = raw_zarr.shape
    if (raw_vz0 < 0 or raw_vy0 < 0 or raw_vx0 < 0 or
            raw_vz1 > raw_shape[0] or raw_vy1 > raw_shape[1] or raw_vx1 > raw_shape[2]):
        raise ValueError(
            f"Crop {crop_id} out of bounds! "
            f"z=[{raw_vz0}:{raw_vz1}] y=[{raw_vy0}:{raw_vy1}] x=[{raw_vx0}:{raw_vx1}] "
            f"vs raw shape {raw_shape}"
        )

    # ---- load raw ----
    raw_crop = raw_zarr[raw_vz0:raw_vz1, raw_vy0:raw_vy1, raw_vx0:raw_vx1].astype(np.uint8)

    # ---- build multi-class label ----
    label_multi = np.zeros((Dz, Dy, Dx), dtype=np.uint8)
    for cname in select_classes:
        s_path = os.path.join(crop_root, cname, scale_name)
        try:
            arr = zarr.open(s_path, mode="r")[:Dz, :Dy, :Dx]
            label_multi[arr > 0] = class_id_map[cname]
        except Exception as e:
            print(f"  Warning: could not load {cname}: {e}")

    print(f"  unique labels: {np.unique(label_multi)}")
    return {"raw": raw_crop, "label": label_multi, "shape": raw_crop.shape, "id": crop_id}


# ---------------------------------------------------------------------------
# Patch dataset
# ---------------------------------------------------------------------------

class PatchSampler3D(Dataset):
    """
    Randomly samples 3D patches from pre-loaded crop volumes.

    Args:
        crop_list:       List of dicts from load_one_crop()
        patch_z/y/x:     Patch size in each dimension
        patches_per_crop: Patches drawn per crop per epoch
        augment:         Whether to apply data augmentation
        num_classes:     Used for label validation
    """

    def __init__(self, crop_list: List[dict],
                 patch_z: int = 128, patch_y: int = 128, patch_x: int = 128,
                 patches_per_crop: int = 50,
                 augment: bool = True,
                 num_classes: int = 10):
        self.crops            = crop_list
        self.patch_z          = patch_z
        self.patch_y          = patch_y
        self.patch_x          = patch_x
        self.patches_per_crop = patches_per_crop
        self.augment          = augment
        self.num_classes      = num_classes

        print(f"PatchSampler3D: {len(crop_list)} crops, "
              f"patch {patch_z}×{patch_y}×{patch_x}, "
              f"{patches_per_crop} patches/crop → "
              f"{len(crop_list) * patches_per_crop} patches/epoch")
        for c in crop_list:
            print(f"  {c['id']}: {c['shape']}")

        self._validate_labels()

    def _validate_labels(self):
        all_labels = set()
        for c in self.crops:
            all_labels.update(np.unique(c['label']).tolist())
        max_label = max(all_labels) if all_labels else 0
        if max_label >= self.num_classes:
            raise RuntimeError(f"Max label {max_label} >= num_classes {self.num_classes}. "
                               "Check SELECT_CLASSES and num_classes in config.")
        print(f"✓ Label check passed: {len(all_labels)} unique classes, max={max_label}\n")

    def __len__(self):
        return len(self.crops) * self.patches_per_crop

    def __getitem__(self, idx):
        crop = self.crops[idx // self.patches_per_crop]

        raw   = crop['raw'].astype(np.float32)
        label = crop['label'].astype(np.int64)
        Dz, Dy, Dx = raw.shape

        # Random patch origin
        z0 = np.random.randint(0, max(1, Dz - self.patch_z + 1))
        y0 = np.random.randint(0, max(1, Dy - self.patch_y + 1))
        x0 = np.random.randint(0, max(1, Dx - self.patch_x + 1))

        raw_p   = raw  [z0:z0+self.patch_z, y0:y0+self.patch_y, x0:x0+self.patch_x]
        label_p = label[z0:z0+self.patch_z, y0:y0+self.patch_y, x0:x0+self.patch_x]

        # Zero-pad if patch extends beyond crop boundary
        if raw_p.shape != (self.patch_z, self.patch_y, self.patch_x):
            raw_p, label_p = self._pad(raw_p, label_p)

        # Augmentation
        if self.augment:
            raw_p, label_p = self._augment(raw_p, label_p)

        raw_p = raw_p / 255.0
        return (torch.from_numpy(raw_p).unsqueeze(0).float(),
                torch.from_numpy(label_p).long())

    def _pad(self, raw_p, label_p):
        pz, py, px = raw_p.shape
        raw_out   = np.zeros((self.patch_z, self.patch_y, self.patch_x), dtype=np.float32)
        label_out = np.zeros((self.patch_z, self.patch_y, self.patch_x), dtype=np.int64)
        raw_out  [:pz, :py, :px] = raw_p
        label_out[:pz, :py, :px] = label_p
        return raw_out, label_out

    def _augment(self, raw_p, label_p):
        # Random axis flips
        for axis in range(3):
            if np.random.rand() > 0.5:
                raw_p   = np.flip(raw_p,   axis=axis).copy()
                label_p = np.flip(label_p, axis=axis).copy()

        # Brightness
        if np.random.rand() > 0.5:
            raw_p = np.clip(raw_p * np.random.uniform(0.85, 1.15), 0, 255)

        # Contrast
        if np.random.rand() > 0.5:
            mean  = raw_p.mean()
            raw_p = np.clip((raw_p - mean) * np.random.uniform(0.85, 1.15) + mean, 0, 255)

        # Small elastic shift (roll per axis)
        if np.random.rand() > 0.3:
            for axis in range(3):
                shift = np.random.randint(-2, 3)
                raw_p   = np.roll(raw_p,   shift, axis=axis)
                label_p = np.roll(label_p, shift, axis=axis)

        return raw_p, label_p
