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


def build_rare_crop_sampling(
    train_labels_dir: str | Path,
    *,
    num_classes: int = 30,
    extra_fraction: float = 0.50,
    sampling_strength: float = 2.0,
    max_tile_repeats: int = 4,
    seed: int = 42,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Return extra per-tile appearances, based ONLY on geographic training masks.

    All original training tiles appear at least once per epoch. Extra appearances
    are drawn with preference for tiles containing scarce crop-class pixels.
    No extra image copies are written; duplicate file paths are passed to GeoAI.
    Class IDs in source masks are 1..num_classes, with 255 ignored.
    """
    if not (0 <= extra_fraction <= max_tile_repeats - 1):
        raise ValueError("extra_fraction must be between 0 and max_tile_repeats - 1")
    if sampling_strength < 0 or max_tile_repeats < 1:
        raise ValueError("sampling_strength must be >= 0 and max_tile_repeats >= 1")

    masks = sorted(Path(train_labels_dir).glob("*.tif"))
    if not masks:
        raise FileNotFoundError(f"No training masks in {train_labels_dir}")

    tile_counts = []
    for path in masks:
        with rasterio.open(path) as src:
            mask = src.read(1)
        valid = (mask >= 1) & (mask <= num_classes)
        if not np.all(valid | (mask == 255)):
            raise ValueError(f"Unexpected crop IDs in {path.name}")
        tile_counts.append(np.bincount(mask[valid].astype(np.int64) - 1,
                                       minlength=num_classes))
    counts = np.stack(tile_counts)
    totals = counts.sum(axis=0)
    present = totals > 0
    rarity = np.ones(num_classes, dtype=np.float64)
    if present.any():
        median_count = np.median(totals[present])
        rarity[present] = np.clip(np.sqrt(median_count / totals[present]), 1.0, 8.0)
    # The value 1 corresponds to common crops; >1 favors rare-class pixels.
    valid_pixels = counts.sum(axis=1)
    fractions = np.divide(counts, valid_pixels[:, None],
                          out=np.zeros_like(counts, dtype=np.float64),
                          where=valid_pixels[:, None] > 0)
    score = fractions @ (rarity - 1.0)
    # Common/empty tiles still have nonzero sampling probability.
    probabilities = 1.0 + sampling_strength * score
    probabilities /= probabilities.sum()

    wanted = int(round(len(masks) * extra_fraction))
    if wanted > len(masks) * (max_tile_repeats - 1):
        raise ValueError("More extra tiles requested than repeat cap permits")
    extra_counts = np.zeros(len(masks), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for _ in range(wanted):
        allowed = extra_counts < (max_tile_repeats - 1)
        p = probabilities * allowed
        p /= p.sum()
        extra_counts[rng.choice(len(masks), p=p)] += 1

    repeats = {path.name: int(extra) for path, extra in zip(masks, extra_counts)}
    summary = {
        "original_tiles": len(masks),
        "extra_tiles": int(extra_counts.sum()),
        "effective_epoch_tiles": len(masks) + int(extra_counts.sum()),
        "max_repeats": int(extra_counts.max() + 1),
        "present_classes": int(present.sum()),
        "rare_classes_1based": (np.where(present)[0][np.argsort(-rarity[present])[:8]] + 1).tolist(),
        "rare_class_weights": np.sort(rarity[present])[-8:][::-1].round(2).tolist(),
    }
    return repeats, summary


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
    rare_crop_sampling: bool = True,
    extra_fraction: float = 0.50,
    sampling_strength: float = 2.0,
    max_tile_repeats: int = 4,
    sampling_seed: int = 42,
    num_epochs: int = 10,
    early_stopping_patience: int | None = 25,
    monitor: str = "val_iou",
    batch_size: int = 3,
    learning_rate: float = 0.001,
    num_workers: int = 0,
    device: Any = None,
    overwrite_model: bool = False,
    plot_curves: bool = True,
) -> Any:
    """Stage train/val tiles, weight classes, optionally oversample rare crops, and train GeoAI U-Net.

    GeoAI monitors validation IoU (higher is better) by default and stops
    after early_stopping_patience epochs without improvement. The best
    validation checkpoint is saved as best_model.pth.


    IMPORTANT: GeoAI normally makes its own random validation split. This
    temporarily patches its train_test_split call with a fixed filename-based
    split. The patch is restored in a finally block. Do not run two trainings
    concurrently in the same Python process.
    """
    if not experiment_name or Path(experiment_name).name != experiment_name:
        raise ValueError("experiment_name must be a simple folder name")

    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1 or None")
    if monitor not in ("val_iou", "val_loss"):
        raise ValueError("monitor must be val_iou or val_loss")

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

    extra_repeats = {}
    if rare_crop_sampling:
        extra_repeats, summary = build_rare_crop_sampling(
            tiles_dir / "train/labels",
            num_classes=num_classes,
            extra_fraction=extra_fraction,
            sampling_strength=sampling_strength,
            max_tile_repeats=max_tile_repeats,
            seed=sampling_seed,
        )
        if set(extra_repeats) != split_names["train"]:
            raise ValueError("Sampling masks do not match staged training tile names")
        print("Rare-crop sampling:", summary)

    training_fn = inspect.unwrap(geoai.train_segmentation_model)
    # Fail fast on older GeoAI versions; do not silently ignore early stopping.
    parameters = inspect.signature(training_fn).parameters
    required = ("early_stopping_patience", "save_best_only")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise RuntimeError(
            "Your installed GeoAI train_segmentation_model does not explicitly "
            f"support {missing}. Upgrade GeoAI or use a compatible version; "
            "training has NOT started."
        )
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
        # Repeat training paths in memory; validation paths are NEVER duplicated.
        # Every original tile stays present at least once.
        expanded_train = []
        for pair in train_pairs:
            expanded_train.extend([pair] * (1 + extra_repeats.get(Path(pair[0]).name, 0)))
        print(f"Fixed geographic split: {len(train_pairs)} unique train tiles "
              f"({len(expanded_train)} appearances/epoch), {len(val_pairs)} val tiles")
        return (
            [p[0] for p in expanded_train],
            [p[0] for p in val_pairs],
            [p[1] for p in expanded_train],
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
            early_stopping_patience=early_stopping_patience,
            save_best_only=True,
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
