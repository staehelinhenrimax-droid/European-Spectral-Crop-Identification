"""Reusable GeoAI crop-segmentation training with geographic train/val separation.

Input tile structure (original Sentinel-2 reflectance and crop IDs)::

    tiles_dir/
      train/images/*.tif  train/labels/*.tif
      val/images/*.tif    val/labels/*.tif
      test/images/*.tif   test/labels/*.tif   # never used here

Images are expected to use Sentinel-2's 0..10000 reflectance scale.
Masks use original crop IDs 1..num_classes and ignore value 255.

Run only in the SDS320 GeoAI/PyTorch environment, not the baseline SDS320 env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect
import re
import shutil

import numpy as np
import rasterio


def prepare_geoai_inputs(
    tiles_dir: str | Path,
    staging_dir: str | Path,
    *,
    num_channels: int,
    num_classes: int = 30,
    reflectance_scale: float = 10000.0,
    overwrite: bool = True,
) -> dict[str, set[str]]:
    """Stage train/val tiles for GeoAI, preserving the held-out test split.

    Converts reflectance to 0..255 float32 because GeoAI scales images by 255;
    converts original crop IDs 1..N to network labels 0..N-1, preserving 255.
    """
    tiles_dir, staging_dir = Path(tiles_dir), Path(staging_dir)
    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_dir}")
    if reflectance_scale <= 0:
        raise ValueError("reflectance_scale must be positive")

    # Validate sources before touching the previously staged data.
    sources: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        image_dir = tiles_dir / split / "images"
        label_dir = tiles_dir / split / "labels"
        images = sorted(image_dir.glob("*.tif"))
        labels = sorted(label_dir.glob("*.tif"))
        if not images or {p.name for p in images} != {p.name for p in labels}:
            raise ValueError(f"Missing/mismatched {split} image/label tiles")
        sources[split] = images

    train_names = {p.name for p in sources["train"]}
    val_names = {p.name for p in sources["val"]}
    if train_names & val_names:
        raise ValueError("Train and validation image filenames overlap")

    if staging_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Staging directory exists: {staging_dir}")
        shutil.rmtree(staging_dir)
    stage_images = staging_dir / "images"
    stage_labels = staging_dir / "labels"
    stage_images.mkdir(parents=True)
    stage_labels.mkdir(parents=True)

    for split, images in sources.items():
        label_dir = tiles_dir / split / "labels"
        for image_path in images:
            with rasterio.open(image_path) as src:
                if src.count != num_channels:
                    raise ValueError(
                        f"{image_path.name}: expected {num_channels} bands, got {src.count}"
                    )
                image = src.read().astype(np.float32)
                image_profile = src.profile.copy()
                image_crs, image_transform = src.crs, src.transform

            with rasterio.open(label_dir / image_path.name) as src:
                if (
                    src.count != 1
                    or src.width != image.shape[2]
                    or src.height != image.shape[1]
                    or src.crs != image_crs
                    or not src.transform.almost_equals(image_transform)
                ):
                    raise ValueError(f"Image and mask grids differ: {image_path.name}")
                labels = src.read(1)
                label_profile = src.profile.copy()

            valid = (labels >= 1) & (labels <= num_classes)
            if not np.all(valid | (labels == 255)):
                raise ValueError(f"Unexpected label values in {image_path.name}")

            image = np.clip(image / reflectance_scale, 0.0, 1.0) * 255.0
            image_profile.update(dtype="float32", count=num_channels)
            with rasterio.open(stage_images / image_path.name, "w", **image_profile) as dst:
                dst.write(image)

            labels = labels.astype(np.uint8, copy=True)
            labels[valid] -= 1
            label_profile.update(dtype="uint8", nodata=255, count=1)
            with rasterio.open(stage_labels / image_path.name, "w", **label_profile) as dst:
                dst.write(labels, 1)

    return {"train": train_names, "val": val_names}


def calculate_crop_weights(
    train_labels_dir: str | Path,
    *,
    num_classes: int = 30,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
) -> tuple[list[float], np.ndarray]:
    """Inverse square-root frequency weights from ORIGINAL-ID train masks only."""
    train_labels_dir = Path(train_labels_dir)
    masks = sorted(train_labels_dir.glob("*.tif"))
    if not masks:
        raise FileNotFoundError(f"No training labels in {train_labels_dir}")

    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    for path in masks:
        with rasterio.open(path) as src:
            mask = src.read(1)
        valid = (mask >= 1) & (mask <= num_classes)
        if not np.all(valid | (mask == 255)):
            raise ValueError(f"Unexpected crop ID in {path.name}")
        pixel_counts += np.bincount(mask[valid].astype(int) - 1, minlength=num_classes)

    present = pixel_counts > 0
    weights = np.ones(num_classes, dtype=np.float64)
    if np.any(present):
        weights[present] = 1.0 / np.sqrt(pixel_counts[present])
        weights[present] /= weights[present].mean()
        weights[present] = np.clip(weights[present], min_weight, max_weight)
    return weights.tolist(), pixel_counts


def train_geoai_unet(
    *,
    project_dir: str | Path,
    experiment_name: str,
    tiles_dir: str | Path,
    num_channels: int,
    num_classes: int = 30,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = "imagenet",
    use_class_weights: bool = True,
    num_epochs: int = 10,
    batch_size: int = 3,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    device: Any = None,
    overwrite_model: bool = False,
    plot_curves: bool = True,
) -> Any:
    """Stage train/val tiles, optionally weight classes, and train GeoAI U-Net.

    IMPORTANT: GeoAI normally makes its own random validation split. This
    temporarily patches its train_test_split call with a fixed filename-based
    split. The patch is restored in a finally block. Do not run two trainings
    concurrently in the same Python process.
    """
    if not experiment_name or Path(experiment_name).name != experiment_name:
        raise ValueError("experiment_name must be a simple folder name")

    import geoai
    import torch

    project_dir, tiles_dir = Path(project_dir), Path(tiles_dir)
    model_dir = project_dir / "models" / experiment_name
    stage_dir = project_dir / "data/processed/geoai_training_input" / experiment_name

    if (model_dir / "best_model.pth").exists() and not overwrite_model:
        raise FileExistsError(
            f"Trained model already exists at {model_dir}. "
            "Choose another experiment_name or set overwrite_model=True."
        )

    split_names = prepare_geoai_inputs(
        tiles_dir, stage_dir, num_channels=num_channels, num_classes=num_classes
    )
    print(f"Geographic split: {len(split_names['train'])} train tiles, "
          f"{len(split_names['val'])} val tiles; 0 test tiles")

    class_weights = None
    if use_class_weights:
        class_weights, pixel_counts = calculate_crop_weights(
            tiles_dir / "train/labels", num_classes=num_classes
        )
        print(f"Classes in training: {(pixel_counts > 0).sum()}/{num_classes}")
        print(f"Class weight range: {min(class_weights):.3f}–{max(class_weights):.3f}")

    training_fn = inspect.unwrap(geoai.train_segmentation_model)
    scope = training_fn.__globals__
    if "train_test_split" not in scope:
        raise RuntimeError("GeoAI split implementation changed; inspect the installed version")
    original_split = scope["train_test_split"]

    def fixed_geographic_split(image_files, label_files, **kwargs):
        if len(image_files) != len(label_files):
            raise ValueError("GeoAI passed unequal numbers of image/label files")
        pairs = list(zip(image_files, label_files))
        names = [Path(image).name for image, _ in pairs]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate image names passed to GeoAI")
        if any(Path(image).name != Path(label).name for image, label in pairs):
            raise ValueError("GeoAI image/label names do not match")
        if set(names) != split_names["train"] | split_names["val"]:
            raise ValueError("Unexpected/missing GeoAI tiles; refuse to random-split")
        train_pairs = [p for p in pairs if Path(p[0]).name in split_names["train"]]
        val_pairs = [p for p in pairs if Path(p[0]).name in split_names["val"]]
        print(f"Fixed geographic split: {len(train_pairs)} train, {len(val_pairs)} val")
        return (
            [p[0] for p in train_pairs],
            [p[0] for p in val_pairs],
            [p[1] for p in train_pairs],
            [p[1] for p in val_pairs],
        )

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    try:
        scope["train_test_split"] = fixed_geographic_split
        return geoai.train_segmentation_model(
            images_dir=str(stage_dir / "images"),
            labels_dir=str(stage_dir / "labels"),
            output_dir=str(model_dir),
            architecture="unet",
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            class_weights=class_weights,
            num_channels=num_channels,
            num_classes=num_classes,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            val_split=0.25,  # Overridden by fixed_geographic_split
            ignore_index=255,
            num_workers=num_workers,
            device=device,
            verbose=True,
            plot_curves=plot_curves,
        )
    finally:
        scope["train_test_split"] = original_split
