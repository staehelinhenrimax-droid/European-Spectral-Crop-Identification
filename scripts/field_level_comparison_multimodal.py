"""Compare GeoAI U-Nets of different input channel counts at EuroCrops field level.

Use matched validation tile footprints and crop masks, but each model must use
its own correctly normalized staged imagery. Test areas are not used.

GeoAI semantic_segmentation is expected to return label indices 0..29.
Training GeoTIFF labels and original area masks use crop IDs 1..30, ignore 255.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

_AREA_RE = re.compile(r"^germany_2023_area_(\d+)_\d+\.tif$")


def _confusion_from_fields(df, num_classes=30):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    if not df.empty:
        y = df.true_crop_id.to_numpy(dtype=int) - 1
        p = df.predicted_crop_id.to_numpy(dtype=int) - 1
        np.add.at(cm, (y, p), 1)
    true = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    hit = np.diag(cm)
    union = true + pred - hit
    iou = np.divide(hit, union, out=np.full(num_classes, np.nan), where=union>0)
    rec = np.divide(hit, true, out=np.full(num_classes, np.nan), where=true>0)
    prec = np.divide(hit, pred, out=np.full(num_classes, np.nan), where=pred>0)
    f1 = np.divide(2*hit, true+pred, out=np.full(num_classes, np.nan), where=(true+pred)>0)
    return cm, iou, prec, rec, f1


def compare_models_by_field(
    *, project_dir, crop_fields, crop_mapping, model_paths,
    staged_images_dir=None, tiles_dir=None, area_masks_dir=None,
    output_dir=None, min_field_pixels=10, num_classes=30,
    run_inference=True, device=None, model_inputs=None,
):
    """Run tile inference (or use cached predictions), deduplicate spatial overlaps,
    rasterize EuroCrops field IDs, and compare original/rare-crop models.

    model_paths: dict label -> path to best_model.pth.
    model_inputs: optional dict label -> {'num_channels': int,
        'staged_images_dir': Path, 'encoder_name': str}.
        Defaults are supplied for baseline, rare_crop, alpha_earth and combined_144band.
    crop_fields: original reduced_crops GeoDataFrame with hcat4_name column.
    crop_mapping: dict crop name -> ID (1..30), as used for creating masks.
    """
    project_dir = Path(project_dir)
    tiles_dir = Path(tiles_dir or project_dir / 'data/processed/tiles_80band')
    area_masks_dir = Path(area_masks_dir or project_dir / 'data/processed/masks_80band')
    staged_images_dir = Path(staged_images_dir or project_dir / 'data/processed/geoai_training_input/unet_80band_rare_crop_earlystop/images')
    output_dir = Path(output_dir or project_dir / 'data/processed/field_level_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    if crop_fields.crs is None or 'hcat4_name' not in crop_fields.columns:
        raise ValueError('crop_fields must have a CRS and hcat4_name column')
    if set(crop_mapping.values()) != set(range(1, num_classes+1)):
        raise ValueError('crop_mapping must match IDs 1..num_classes from the original masks')
    if min_field_pixels < 1:
        raise ValueError('min_field_pixels must be >= 1')
    models = {str(k): Path(v) for k,v in model_paths.items()}
    if len(models) < 2:
        raise ValueError('Provide two model_paths (baseline and rare_crop)')
    for name,p in models.items():
        if not p.is_file():
            raise FileNotFoundError(f'{name} checkpoint not found: {p}')

    val_tile_names = sorted(p.name for p in (tiles_dir/'val/images').glob('*.tif'))
    if not val_tile_names:
        raise FileNotFoundError('No validation image tiles found')
    val_label_names = {p.name for p in (tiles_dir/'val/labels').glob('*.tif')}
    if set(val_tile_names) != val_label_names:
        raise ValueError('Validation image/label files do not match')
    # Distinct inputs are essential: the 64-/144-band models cannot use
    # the original 80-band reflectance imagery.
    defaults = {
        'baseline': (80, project_dir / 'data/processed/geoai_training_input/unet_80band_monthly_weighted_250epochs/images'),
        'rare_crop': (80, project_dir / 'data/processed/geoai_training_input/unet_80band_rare_crop_earlystop/images'),
        'alpha_earth': (64, project_dir / 'data/processed/geoai_training_input/unet_alphaearth_64band/images'),
        'combined_144band': (144, project_dir / 'data/processed/geoai_training_input/unet_combined_144band/images'),
    }
    model_inputs = model_inputs or {}
    configs = {}
    for model_name in models:
        default_channels, default_dir = defaults.get(model_name, (80, staged_images_dir))
        options = model_inputs.get(model_name, {})
        configs[model_name] = {
            'num_channels': int(options.get('num_channels', default_channels)),
            'staged_images_dir': Path(options.get('staged_images_dir', default_dir)),
            'encoder_name': options.get('encoder_name', 'resnet18'),
        }

    by_area = defaultdict(list)
    for name in val_tile_names:
        match = _AREA_RE.match(name)
        if not match:
            raise ValueError(f'Unexpected tile filename: {name}')
        by_area[int(match.group(1))].append(name)
    print(f'Validation: {len(val_tile_names)} tiles across {len(by_area)} geographic areas')

    if run_inference:
        import torch
        import geoai
        if device is None:
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        for model_name, path in models.items():
            pred_dir = output_dir / 'tile_predictions' / model_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            todo = [name for name in val_tile_names if not (pred_dir/name).is_file()]
            print(f'{model_name}: {len(todo)} tiles to predict; {len(val_tile_names)-len(todo)} cached')
            config = configs[model_name]
            if todo:
                image_dir = config['staged_images_dir']
                missing = [name for name in todo if not (image_dir/name).is_file()]
                if missing:
                    raise FileNotFoundError(
                        f'{model_name}: {len(missing)} staged images missing in {image_dir}; '
                        f'first missing: {missing[0]}'
                    )
                with rasterio.open(image_dir/todo[0]) as sample:
                    if sample.count != config['num_channels']:
                        raise ValueError(
                            f"{model_name}: images have {sample.count} bands, "
                            f"configured for {config['num_channels']}"
                        )
                print(f"  Using {config['num_channels']} bands from {image_dir}")
            for i, name in enumerate(todo, 1):
                geoai.semantic_segmentation(
                    input_path=str(config['staged_images_dir']/name),
                    output_path=str(pred_dir/name),
                    model_path=str(path),
                    architecture='unet', encoder_name=config['encoder_name'],
                    num_channels=config['num_channels'], num_classes=num_classes,
                    window_size=128, overlap=0, batch_size=1,
                    device=device, quiet=True,
                )
                if i % 100 == 0 or i == len(todo):
                    print(f'  {model_name}: {i}/{len(todo)} generated')
    else:
        for model_name in models:
            pdir = output_dir/'tile_predictions'/model_name
            missing = [n for n in val_tile_names if not (pdir/n).is_file()]
            if missing:
                raise FileNotFoundError(f'{model_name}: {len(missing)} predictions missing; run_inference=True')

    # Stable identifiers even when the same original field intersects two areas.
    fields = crop_fields.loc[crop_fields.hcat4_name.isin(crop_mapping)].copy().reset_index(drop=True)
    fields['field_id'] = np.arange(1, len(fields)+1, dtype=np.int32)
    projected = {}
    # Accumulate predictions at the field level across areas. Rows are unique field IDs.
    votes_per_model = {name: defaultdict(lambda: np.zeros(num_classes, dtype=np.int64)) for name in models}
    true_ids = {}
    valid_pixels = defaultdict(int)
    area_ids_seen = defaultdict(set)

    for ai,(area_id,tile_names) in enumerate(sorted(by_area.items()),1):
        mask_path = area_masks_dir/f'germany_mask_{area_id}.tif'
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        with rasterio.open(mask_path) as src:
            truth = src.read(1)
            transform, crs = src.transform, src.crs
            height,width = src.height,src.width
            footprint = box(*src.bounds)
        crs_key = crs.to_string()
        if crs_key not in projected:
            projected[crs_key] = fields.to_crs(crs)
        fproj = projected[crs_key]
        candidates = fproj.loc[fproj.geometry.intersects(footprint)].copy()
        candidates = candidates.loc[candidates.geometry.notna() & ~candidates.geometry.is_empty]
        if candidates.empty:
            raise RuntimeError(f'No crop field polygons in validation area {area_id}')
        # Field polygons are clipped only for rasterizing into this area's grid.
        raster_shapes = [
            (geom.intersection(footprint), int(fid))
            for geom,fid in zip(candidates.geometry, candidates.field_id)
        ]
        field_grid = rasterize(
            raster_shapes, out_shape=(height,width), transform=transform,
            fill=0, dtype='int32', all_touched=False,
        )
        # Use only pixels deemed reliable by the ORIGINAL crop mask. This
        # automatically excludes polygon overlaps and 5m field-boundary buffers.
        valid_field = (field_grid > 0) & (truth >= 1) & (truth <= num_classes)
        field_to_crop = dict(zip(candidates.field_id.astype(int), candidates.hcat4_name.map(crop_mapping).astype(int)))
        expected_grid = rasterize(
            [(geom, int(crop_mapping[name])) for geom,name in zip(candidates.geometry, candidates.hcat4_name)],
            out_shape=(height,width), transform=transform, fill=0, dtype="uint8", all_touched=False,
        )
        valid_field &= (truth == expected_grid)
        relevant_field_ids = np.unique(field_grid[valid_field])

        # For every pixel covered by >1 tile, vote across tile predictions and
        # select a SINGLE crop class. Thus overlap never multiplies field count.
        for model_name in models:
            counts = np.zeros((num_classes, height, width), dtype=np.uint16)
            pdir = output_dir/'tile_predictions'/model_name
            for tile_name in tile_names:
                with rasterio.open(tiles_dir/'val/images'/tile_name) as tile:
                    col_float,row_float = (~transform)*(tile.bounds.left, tile.bounds.top)
                    col,row = int(round(col_float)),int(round(row_float))
                    if abs(col_float-col)>0.02 or abs(row_float-row)>0.02:
                        raise ValueError(f'Tile not grid-aligned to area mask: {tile_name}')
                    h,w=tile.height,tile.width
                    if tile.crs != crs or col<0 or row<0 or col+w>width or row+h>height:
                        raise ValueError(f'Tile out of bounds or CRS mismatch: {tile_name}')
                with rasterio.open(pdir/tile_name) as pred_src:
                    pred=pred_src.read(1)
                if pred.shape!=(h,w) or np.any((pred<0)|(pred>=num_classes)):
                    raise ValueError(f'Unexpected prediction values/shape in {pdir/tile_name}; expected 0..{num_classes-1}')
                yy,xx=np.indices((h,w))
                counts[pred, row+yy, col+xx] += 1
            coverage = counts.sum(axis=0)>0
            use = valid_field & coverage
            # One predicted crop per geographic pixel, then one prediction per field.
            winner = counts.argmax(axis=0)+1
            field_ids = field_grid[use].astype(np.int64)
            predicted = winner[use].astype(np.int64)
            # Fast per-field/class contingency without allocating enormous global matrices
            if len(field_ids):
                flat = field_ids*(num_classes+1)+predicted
                keys, cnts = np.unique(flat, return_counts=True)
                for k, n in zip(keys, cnts):
                    fid = int(k//(num_classes+1))
                    crop_id = int(k%(num_classes+1))
                    votes_per_model[model_name][fid][crop_id-1] += int(n)
                if model_name == next(iter(models)):
                    ids, nums = np.unique(field_ids, return_counts=True)
                    for fid,n in zip(ids,nums):
                        fid=int(fid)
                        true_ids[fid]=field_to_crop[fid]
                        valid_pixels[fid]+=int(n)
                        area_ids_seen[fid].add(area_id)
            del counts
        print(f'[{ai}/{len(by_area)}] area {area_id}: {len(relevant_field_ids)} mapped fields')

    model_rows=[]
    for model_name in models:
        for fid, true_crop in true_ids.items():
            votes=votes_per_model[model_name][fid]
            n=int(votes.sum())
            if n < min_field_pixels:
                continue
            predicted=int(votes.argmax()+1)
            model_rows.append(dict(
                model=model_name, field_id=fid,
                crop_name=next(k for k,v in crop_mapping.items() if v==true_crop),
                true_crop_id=true_crop, predicted_crop_id=predicted,
                correct=(predicted==true_crop),
                pixels=n, field_vote_share=float(votes.max()/n),
                area_ids=','.join(map(str,sorted(area_ids_seen[fid]))),
            ))
    detail=pd.DataFrame(model_rows)
    if detail.empty:
        raise RuntimeError('No evaluable fields; check polygons, masks, and minimum pixels')
    detail.to_csv(output_dir/'field_predictions_all_models.csv',index=False)
    per_class=[]
    summary=[]
    inv_map={v:k for k,v in crop_mapping.items()}
    for model_name in models:
        subset=detail.loc[detail.model==model_name]
        cm,iou,precision,recall,f1=_confusion_from_fields(subset,num_classes)
        for i in range(num_classes):
            per_class.append(dict(model=model_name,crop_id=i+1,crop=inv_map[i+1],
                true_fields=int(cm.sum(axis=1)[i]),predicted_fields=int(cm.sum(axis=0)[i]),
                correct_fields=int(cm[i,i]),field_IoU=iou[i],field_precision=precision[i],
                field_recall=recall[i],field_F1=f1[i]))
        summary.append(dict(model=model_name,evaluated_fields=len(subset),
            field_accuracy=float(subset.correct.mean()),
            macro_field_IoU=float(np.nanmean(iou)),
            macro_field_F1=float(np.nanmean(f1)),
            mean_field_vote_share=float(subset.field_vote_share.mean())))
    by_class=pd.DataFrame(per_class)
    overview=pd.DataFrame(summary)
    by_class.to_csv(output_dir/'field_metrics_by_crop.csv',index=False)
    overview.to_csv(output_dir/'field_metric_summary.csv',index=False)
    print('\nFIELD-LEVEL RESULTS (each physical field counts once)')
    print(overview.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    pivot = by_class.pivot(index=['crop_id','crop'],columns='model',values='field_recall')
    print('\nPER-CROP FIELD RECALL (fraction of fields correctly identified):')
    print(pivot.round(3).to_string())
    print(f'\nSaved CSVs to {output_dir}')
    return detail, by_class, overview
