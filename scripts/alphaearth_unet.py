"""64-band AlphaEarth U-Net preparation and training for SDS320.

Uses the validated 2023 AlphaEarth area rasters and the SAME 128x128 footprints,
labels, and geographic train/validation/test split as tiles_80band.

Only run training from the sds320-geoai-test Python environment.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import inspect

import numpy as np
import rasterio
from rasterio.windows import Window

NAME_RE = re.compile(r"germany_2023_area_(\d+)_\d+\.tif$")


def create_alphaearth_tiles(
    project_dir,
    source_tiles_dir=None,
    alphaearth_images_dir=None,
    output_tiles_dir=None,
    overwrite=False,
):
    """Write 64-band tiles corresponding exactly to every original 80-band tile.

    Original labels (IDs 1..30, 255 ignored) are copied unchanged. No overlapping
    geographic splits are introduced, and original data is never altered.
    """
    project_dir = Path(project_dir)
    source_tiles_dir = Path(source_tiles_dir or project_dir / 'data/processed/tiles_80band')
    alphaearth_images_dir = Path(alphaearth_images_dir or project_dir / 'data/processed/alphaearth_2023/images')
    output_tiles_dir = Path(output_tiles_dir or project_dir / 'data/processed/tiles_alphaearth_64band')

    jobs = []
    for split in ('train', 'val', 'test'):
        images_dir = source_tiles_dir / split / 'images'
        labels_dir = source_tiles_dir / split / 'labels'
        imgs = sorted(images_dir.glob('*.tif'))
        labs = {p.name for p in labels_dir.glob('*.tif')}
        if not imgs or {p.name for p in imgs} != labs:
            raise ValueError(f'Missing or unmatched {split} images/labels')
        for p in imgs:
            match = NAME_RE.fullmatch(p.name)
            if not match:
                raise ValueError(f'Unexpected tile filename: {p.name}')
            ae_path = alphaearth_images_dir / f'germany_2023_area_{match.group(1)}.tif'
            if not ae_path.exists():
                raise FileNotFoundError(f'Missing validated AlphaEarth raster: {ae_path}')
            jobs.append((split, p, labels_dir / p.name, ae_path))
    if output_tiles_dir.exists():
        if not overwrite:
            raise FileExistsError(f'{output_tiles_dir} already exists. Set overwrite=True only if you intend to recreate it.')
        shutil.rmtree(output_tiles_dir)
    for split in ('train', 'val', 'test'):
        (output_tiles_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_tiles_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    count = {'train': 0, 'val': 0, 'test': 0}
    # Opening one AE file per tile is less memory-intensive than caching all images.
    for split, tile_path, label_path, ae_path in jobs:
        with rasterio.open(tile_path) as tile, rasterio.open(ae_path) as ae:
            if tile.count != 80 or ae.count != 64 or ae.crs != tile.crs:
                raise ValueError(f'Unexpected bands/CRS: {tile_path.name}')
            if not tile.transform.is_rectilinear or not ae.transform.is_rectilinear:
                raise ValueError('Rotated reference grids are not supported')
            if abs(ae.res[0] - tile.res[0]) > 1e-5 or abs(ae.res[1] - tile.res[1]) > 1e-5:
                raise ValueError(f'Pixel resolution differs: {tile_path.name}')
            col, row = (~ae.transform) * (tile.transform.c, tile.transform.f)
            c0, r0 = round(col), round(row)
            if abs(col - c0) > 1e-4 or abs(row - r0) > 1e-4:
                raise ValueError(f'Tile is not aligned to AE pixel grid: {tile_path.name}')
            if (c0 < 0 or r0 < 0 or c0 + tile.width > ae.width or
                    r0 + tile.height > ae.height):
                raise ValueError(f'Tile exceeds AlphaEarth area: {tile_path.name}')
            window = Window(c0, r0, tile.width, tile.height)
            if not ae.window_transform(window).almost_equals(tile.transform):
                raise ValueError(f'Tile grid transform mismatch: {tile_path.name}')
            values = ae.read(window=window, out_dtype='float32')
            if not np.isfinite(values).all() or (values == -9999).any():
                raise ValueError(f'Missing/invalid AlphaEarth values: {tile_path.name}')
            if values.min() < -1.05 or values.max() > 1.05:
                raise ValueError(f'Unexpected AlphaEarth range [{values.min()}, {values.max()}] in {tile_path.name}')
            profile = tile.profile.copy()
            profile.update(driver='GTiff', count=64, dtype='float32', nodata=None, compress='deflate')
            output = output_tiles_dir / split / 'images' / tile_path.name
            with rasterio.open(output, 'w', **profile) as dst:
                dst.write(values)
        shutil.copy2(label_path, output_tiles_dir / split / 'labels' / label_path.name)
        count[split] += 1
    print('Created exactly matching 64-band AlphaEarth tiles:', count)
    print('Saved to:', output_tiles_dir)
    return output_tiles_dir


def train_alphaearth_unet(
    *, project_dir, tiles_dir=None,
    experiment_name='unet_alphaearth_64band',
    num_epochs=150, early_stopping_patience=25, monitor='val_iou',
    batch_size=2, learning_rate=0.001, use_class_weights=True,
    num_workers=0, encoder_name='resnet18', encoder_weights='imagenet',
    overwrite_model=False, plot_curves=True,
):
    """Reuse our existing GeoAI training wrapper, fixing only AlphaEarth scaling.

    The GeoAI loader divides staged data by 255. For embedding input [-1,1],
    stage 127.5*(embedding+1), yielding values in [0,1] after GeoAI /255.
    Validation split stays geographic; no minority-crop oversampling.
    """
    import scripts.geoai_training_rare_crop as base

    project_dir = Path(project_dir)
    tiles_dir = Path(tiles_dir or project_dir / 'data/processed/tiles_alphaearth_64band')
    scope = base.train_geoai_unet.__globals__
    original_prepare = scope['prepare_geoai_inputs']

    def prepare_alphaearth_inputs(tiles_dir, staging_dir, *, num_channels, num_classes=30):
        import shutil
        tiles_dir, staging_dir = Path(tiles_dir), Path(staging_dir)
        if num_channels != 64:
            raise ValueError('AlphaEarth expects exactly 64 channels')
        sources = {}
        for split in ('train', 'val'):
            imgs = sorted((tiles_dir / split / 'images').glob('*.tif'))
            labels = {p.name for p in (tiles_dir / split / 'labels').glob('*.tif')}
            if not imgs or {p.name for p in imgs} != labels:
                raise ValueError(f'Missing/unmatched {split} image-label tiles')
            sources[split] = imgs
        train_names = {p.name for p in sources['train']}
        val_names = {p.name for p in sources['val']}
        if train_names & val_names:
            raise ValueError('Geographic split filenames overlap')
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        (staging_dir / 'images').mkdir(parents=True)
        (staging_dir / 'labels').mkdir(parents=True)
        for split, imgs in sources.items():
            for image_path in imgs:
                with rasterio.open(image_path) as src:
                    if src.count != 64:
                        raise ValueError(f'Expected 64 channels in {image_path.name}')
                    values = src.read(out_dtype='float32')
                    profile = src.profile.copy()
                    image_crs, image_transform = src.crs, src.transform
                label_path = tiles_dir / split / 'labels' / image_path.name
                with rasterio.open(label_path) as src:
                    labels = src.read(1)
                    label_profile = src.profile.copy()
                    if (src.count != 1 or src.width != values.shape[2]
                        or src.height != values.shape[1] or src.crs != image_crs
                        or not src.transform.almost_equals(image_transform)):
                        raise ValueError(f'Misaligned label: {image_path.name}')
                valid = (labels >= 1) & (labels <= num_classes)
                if not np.all(valid | (labels == 255)):
                    raise ValueError(f'Unexpected labels in {image_path.name}')
                if (not np.isfinite(values).all() or (values == -9999).any()
                    or values.min() < -1.05 or values.max() > 1.05):
                    raise ValueError(f'Invalid AlphaEarth embeddings: {image_path.name}')
                # GeoAI divides images by 255: this maps [-1,1] to [0,1]
                # without discarding negative embedding coordinates.
                values = np.clip((values + 1.0) * 127.5, 0, 255).astype('float32')
                profile.update(driver='GTiff', count=64, dtype='float32', nodata=None)
                with rasterio.open(staging_dir / 'images' / image_path.name, 'w', **profile) as dst:
                    dst.write(values)
                labels = labels.astype('uint8', copy=True)
                labels[valid] -= 1
                label_profile.update(driver='GTiff', dtype='uint8', count=1, nodata=255)
                with rasterio.open(staging_dir / 'labels' / image_path.name, 'w', **label_profile) as dst:
                    dst.write(labels, 1)
        return {'train': train_names, 'val': val_names}

    if scope['prepare_geoai_inputs'] is not original_prepare:
        raise RuntimeError('GeoAI preparation was already patched. Restart kernel and retry.')
    try:
        scope['prepare_geoai_inputs'] = prepare_alphaearth_inputs
        return base.train_geoai_unet(
            project_dir=project_dir,
            experiment_name=experiment_name,
            tiles_dir=tiles_dir,
            num_channels=64,
            num_classes=30,
            rare_crop_sampling=False,
            use_class_weights=use_class_weights,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            monitor=monitor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_workers=num_workers,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            overwrite_model=overwrite_model,
            plot_curves=plot_curves,
        )
    finally:
        scope['prepare_geoai_inputs'] = original_prepare
