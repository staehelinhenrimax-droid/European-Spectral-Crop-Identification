"""144-band (80 monthly Sentinel-2 + 64 annual AlphaEarth) U-Net workflow.

Uses *existing* matched 80-band and AlphaEarth 64-band tile sets.
Original tiles and masks are never modified.

IMPORTANT: A special GeoAI staging step normalizes each channel group separately:
    Sentinel-2 reflectance 0..10000 -> 0..255
    AlphaEarth embedding -1..1   -> 0..255
GeoAI divides staged imagery by 255 during training and inference.
For inference use images from the experiment-specific staged directory.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import rasterio


def create_combined_tiles(
    project_dir,
    s2_tiles_dir=None,
    ae_tiles_dir=None,
    output_tiles_dir=None,
    overwrite=False,
):
    project_dir = Path(project_dir)
    s2_tiles_dir = Path(s2_tiles_dir or project_dir / 'data/processed/tiles_80band')
    ae_tiles_dir = Path(ae_tiles_dir or project_dir / 'data/processed/tiles_alphaearth_64band')
    output_tiles_dir = Path(output_tiles_dir or project_dir / 'data/processed/tiles_combined_144band')

    jobs = []
    for split in ('train', 'val', 'test'):
        s2_images = s2_tiles_dir / split / 'images'
        ae_images = ae_tiles_dir / split / 'images'
        s2_labels = s2_tiles_dir / split / 'labels'
        ae_labels = ae_tiles_dir / split / 'labels'
        names = {p.name for p in s2_images.glob('*.tif')}
        if not names:
            raise ValueError(f'No {split} Sentinel-2 tiles found in {s2_images}')
        for directory in (ae_images, s2_labels, ae_labels):
            other = {p.name for p in directory.glob('*.tif')}
            if names != other:
                raise ValueError(f'{split} filenames differ in {directory}: '
                                 f'{len(names - other)} missing, {len(other - names)} extra')
        jobs.extend((split, name) for name in sorted(names))

    if output_tiles_dir.exists():
        if not overwrite:
            raise FileExistsError(f'{output_tiles_dir} already exists; '
                                  'set overwrite=True to regenerate (deletes only that directory)')
        shutil.rmtree(output_tiles_dir)
    for split in ('train', 'val', 'test'):
        (output_tiles_dir / split / 'images').mkdir(parents=True)
        (output_tiles_dir / split / 'labels').mkdir(parents=True)

    counts = dict.fromkeys(('train', 'val', 'test'), 0)
    for split, name in jobs:
        s2_path = s2_tiles_dir / split / 'images' / name
        ae_path = ae_tiles_dir / split / 'images' / name
        s2_label = s2_tiles_dir / split / 'labels' / name
        ae_label = ae_tiles_dir / split / 'labels' / name
        with rasterio.open(s2_path) as src_s2, rasterio.open(ae_path) as src_ae, \
             rasterio.open(s2_label) as lbl_s2, rasterio.open(ae_label) as lbl_ae:
            if src_s2.count != 80 or src_ae.count != 64:
                raise ValueError(f'{name}: expected 80 and 64 bands')
            grid = (src_s2.crs == src_ae.crs and src_s2.width == src_ae.width
                    and src_s2.height == src_ae.height
                    and src_s2.transform.almost_equals(src_ae.transform))
            label_grids = all(
                x.crs == src_s2.crs and x.width == src_s2.width
                and x.height == src_s2.height and x.transform.almost_equals(src_s2.transform)
                for x in (lbl_s2, lbl_ae)
            )
            if not grid or not label_grids:
                raise ValueError(f'{name}: spatial grid mismatch')
            original_labels = lbl_s2.read(1)
            if not np.array_equal(original_labels, lbl_ae.read(1)):
                raise ValueError(f'{name}: AlphaEarth and Sentinel-2 labels differ')
            s2 = src_s2.read(out_dtype='float32')
            ae = src_ae.read(out_dtype='float32')
            if not np.isfinite(s2).all() or not np.isfinite(ae).all():
                raise ValueError(f'{name}: nonfinite pixel values')
            if ae.min() < -1.05 or ae.max() > 1.05:
                raise ValueError(f'{name}: unexpected AlphaEarth range {ae.min()}, {ae.max()}')
            profile = src_s2.profile.copy()
            profile.update(driver='GTiff', count=144, dtype='float32', nodata=None, compress='deflate')
            with rasterio.open(output_tiles_dir / split / 'images' / name, 'w', **profile) as dst:
                dst.write(np.concatenate((s2, ae), axis=0))
            # Same original labels: crop IDs 1..30 and ignore 255.
            shutil.copy2(s2_label, output_tiles_dir / split / 'labels' / name)
        counts[split] += 1
    print('144-band tiles created:', counts)
    print('Saved to:', output_tiles_dir)
    return output_tiles_dir


def train_combined_unet(
    *, project_dir,
    tiles_dir=None,
    experiment_name='unet_combined_144band',
    num_epochs=150,
    early_stopping_patience=25,
    monitor='val_iou',
    batch_size=2,
    learning_rate=0.001,
    use_class_weights=True,
    num_workers=0,
    encoder_name='resnet18',
    encoder_weights='imagenet',
    overwrite_model=False,
    plot_curves=True,
    num_classes = 30
):
    """Train with correct separate band scaling and GeoAI geographic holdouts."""
    import scripts.geoai_training_rare_crop as base

    project_dir = Path(project_dir)
    tiles_dir = Path(tiles_dir or project_dir / 'data/processed/tiles_combined_144band')
    scope = base.train_geoai_unet.__globals__
    original_prepare = scope['prepare_geoai_inputs']

    def prepare_combined_inputs(tiles_dir, staging_dir, *, num_channels, num_classes=30):
        tiles_dir, staging_dir = Path(tiles_dir), Path(staging_dir)
        if num_channels != 144:
            raise ValueError('Combined U-Net requires 144 input channels')
        sources = {}
        for split in ('train', 'val'):
            imgs = sorted((tiles_dir / split / 'images').glob('*.tif'))
            labels = {p.name for p in (tiles_dir / split / 'labels').glob('*.tif')}
            if not imgs or {p.name for p in imgs} != labels:
                raise ValueError(f'Unmatched {split} images/labels')
            sources[split] = imgs
        train_names = {p.name for p in sources['train']}
        val_names = {p.name for p in sources['val']}
        if train_names & val_names:
            raise ValueError('Training and validation filenames overlap')
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        (staging_dir / 'images').mkdir(parents=True)
        (staging_dir / 'labels').mkdir(parents=True)

        for split, images in sources.items():
            for img_path in images:
                with rasterio.open(img_path) as src:
                    if src.count != 144:
                        raise ValueError(f'{img_path.name}: expected 144 bands')
                    features = src.read(out_dtype='float32')
                    profile = src.profile.copy()
                    image_crs, image_transform = src.crs, src.transform
                with rasterio.open(tiles_dir / split / 'labels' / img_path.name) as src:
                    labels = src.read(1)
                    label_profile = src.profile.copy()
                    if (src.count != 1 or src.width != features.shape[2]
                        or src.height != features.shape[1] or src.crs != image_crs
                        or not src.transform.almost_equals(image_transform)):
                        raise ValueError(f'{img_path.name}: image/label grid mismatch')
                valid = (labels >= 1) & (labels <= num_classes)
                if not np.all(valid | (labels == 255)):
                    raise ValueError(f'{img_path.name}: invalid crop labels')
                if not np.isfinite(features).all():
                    raise ValueError(f'{img_path.name}: nonfinite features')
                if features[80:].min() < -1.05 or features[80:].max() > 1.05:
                    raise ValueError(f'{img_path.name}: AlphaEarth values outside expected range')

                # GeoAI divides staged float images by 255. Same transformations
                # as the original 80-band and 64-band models, respectively.
                features[:80] = np.clip(features[:80] / 10000.0, 0, 1) * 255.0
                features[80:] = np.clip((features[80:] + 1.0) * 127.5, 0, 255)
                profile.update(driver='GTiff', count=144, dtype='float32', nodata=None, compress='deflate')
                with rasterio.open(staging_dir / 'images' / img_path.name, 'w', **profile) as dst:
                    dst.write(features)
                labels = labels.astype('uint8', copy=True)
                labels[valid] -= 1
                label_profile.update(driver='GTiff', count=1, dtype='uint8', nodata=255)
                with rasterio.open(staging_dir / 'labels' / img_path.name, 'w', **label_profile) as dst:
                    dst.write(labels, 1)
        return {'train': train_names, 'val': val_names}

    if scope['prepare_geoai_inputs'] is not original_prepare:
        raise RuntimeError('Preprocessing already patched: restart kernel')
    try:
        scope['prepare_geoai_inputs'] = prepare_combined_inputs
        return base.train_geoai_unet(
            project_dir=project_dir,
            experiment_name=experiment_name,
            tiles_dir=tiles_dir,
            num_channels=144,
            num_classes=num_classes,
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
