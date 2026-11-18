"""Field-level Random Forest on 2023 AlphaEarth embeddings.

Uses the same EuroCrops polygons, 30 crop IDs, 5 m boundary-masked
area truth rasters and *geographic* train/val split as the existing U-Net.
No test fields are loaded or evaluated.

Input AlphaEarth TIFFs: data/processed/alphaearth_2023/images/germany_2023_area_<id>.tif
Source mask TIFFs:    data/processed/masks_80band/germany_mask_<id>.tif
Tile split:           data/processed/tiles_80band/{train,val}/images/*.tif
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

_TILE_RE = re.compile(r"^germany_2023_area_(\d+)_\d+\.tif$")


def _area_ids(tiles_dir: Path, split: str) -> set[int]:
    paths = sorted((tiles_dir / split / "images").glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No {split} tiles at {tiles_dir / split / 'images'}")
    ids = set()
    for p in paths:
        match = _TILE_RE.fullmatch(p.name)
        if match is None:
            raise ValueError(f"Unexpected tile name: {p.name}")
        ids.add(int(match.group(1)))
    return ids


def extract_field_features(*, project_dir, crop_fields, crop_mapping,
                           alphaearth_dir=None, area_masks_dir=None,
                           tiles_dir=None, min_field_pixels=10):
    """One row per physical EuroCrops field (64 median AlphaEarth features).

    Exactly matches the earlier field evaluator's polygon rasterization, crop
    mask exclusion and original field IDs. A rare field spanning multiple areas
    receives pixel-count-weighted mean of its per-area feature medians.
    """
    root = Path(project_dir)
    alphaearth_dir = Path(alphaearth_dir or root / "data/processed/alphaearth_2023/images")
    area_masks_dir = Path(area_masks_dir or root / "data/processed/masks_80band")
    tiles_dir = Path(tiles_dir or root / "data/processed/tiles_80band")
    if crop_fields.crs is None or "hcat4_name" not in crop_fields:
        raise ValueError("crop_fields must have a CRS and hcat4_name column")
    if set(crop_mapping.values()) != set(range(1, 31)):
        raise ValueError("crop_mapping must contain the original IDs 1..30")
    if min_field_pixels < 1:
        raise ValueError("min_field_pixels must be >= 1")

    ids_train, ids_val = _area_ids(tiles_dir, "train"), _area_ids(tiles_dir, "val")
    if ids_train & ids_val:
        raise ValueError("Geographic train/val leakage: overlapping area IDs")
    print(f"Areas: {len(ids_train)} train, {len(ids_val)} validation (test excluded)")

    fields = crop_fields.loc[crop_fields.hcat4_name.isin(crop_mapping)].copy().reset_index(drop=True)
    fields["field_id"] = np.arange(1, len(fields) + 1, dtype=np.int32)
    projected = {}
    sums = defaultdict(lambda: np.zeros(64, dtype=np.float64))
    counts = defaultdict(int)
    labels = {}
    per_field_split = {}
    per_field_areas = defaultdict(set)

    for pos, (area_id, split) in enumerate(
        [(a, "train") for a in sorted(ids_train)] + [(a, "val") for a in sorted(ids_val)], 1
    ):
        img_path = alphaearth_dir / f"germany_2023_area_{area_id}.tif"
        mask_path = area_masks_dir / f"germany_mask_{area_id}.tif"
        if not img_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing AlphaEarth image or mask: {img_path}, {mask_path}")
        with rasterio.open(img_path) as src, rasterio.open(mask_path) as ground:
            if src.count != 64 or ground.count != 1:
                raise ValueError(f"Area {area_id}: expected 64 embedding bands and one truth band")
            if (src.crs != ground.crs or src.width != ground.width or
                src.height != ground.height or not src.transform.almost_equals(ground.transform)):
                raise ValueError(f"Area {area_id}: AlphaEarth and original crop-mask grids differ")
            features = src.read().astype(np.float32)
            truth = ground.read(1)
            crs, transform, footprint = src.crs, src.transform, box(*src.bounds)
        key = crs.to_string()
        if key not in projected:
            projected[key] = fields.to_crs(crs)
        fproj = projected[key]
        candidates = fproj.loc[fproj.geometry.intersects(footprint)].copy()
        candidates = candidates.loc[candidates.geometry.notna() & ~candidates.geometry.is_empty]
        if candidates.empty:
            raise RuntimeError(f"Area {area_id}: no EuroCrops fields overlap")
        shapes = [(geom.intersection(footprint), int(fid))
                  for geom, fid in zip(candidates.geometry, candidates.field_id)]
        field_grid = rasterize(shapes, out_shape=truth.shape, transform=transform,
                               fill=0, dtype="int32", all_touched=False)
        expected = rasterize(
            [(geom, int(crop_mapping[name])) for geom, name in
             zip(candidates.geometry, candidates.hcat4_name)],
            out_shape=truth.shape, transform=transform,
            fill=0, dtype="uint8", all_touched=False)
        # Match field_level_comparison.py exactly: ignore unknown, overlapping,
        # boundary-buffered pixels and disagreeing polygon/mask assignments.
        valid = (field_grid > 0) & (truth >= 1) & (truth <= 30) & (truth == expected)
        valid &= np.all(np.isfinite(features) & (features != -9999), axis=0)
        field_to_class = dict(zip(candidates.field_id.astype(int),
                                  candidates.hcat4_name.map(crop_mapping).astype(int)))
        used = 0
        for fid in np.unique(field_grid[valid]):
            fid = int(fid)
            pixels = valid & (field_grid == fid)
            n = int(pixels.sum())
            if n == 0:
                continue
            if fid in per_field_split and per_field_split[fid] != split:
                raise ValueError(f"Physical field {fid} crosses train/val split; aborting")
            values = features[:, pixels]
            medians = np.median(values, axis=1)
            sums[fid] += medians.astype(np.float64) * n
            counts[fid] += n
            labels[fid] = field_to_class[fid]
            per_field_split[fid] = split
            per_field_areas[fid].add(area_id)
            used += 1
        print(f"[{pos}/{len(ids_train) + len(ids_val)}] {split} area {area_id}: {used} labeled fields")

    names = [f"A{i:02d}" for i in range(64)]
    inv_map = {v: k for k, v in crop_mapping.items()}
    records = []
    for fid in sorted(counts):
        if counts[fid] < min_field_pixels:
            continue
        feats = sums[fid] / counts[fid]
        record = dict(field_id=fid, split=per_field_split[fid],
                      true_crop_id=labels[fid], crop=inv_map[labels[fid]],
                      pixels=counts[fid],
                      area_ids=",".join(map(str, sorted(per_field_areas[fid]))))
        record.update(zip(names, feats))
        records.append(record)
    df = pd.DataFrame.from_records(records)
    if df.empty or df.split.nunique() != 2:
        raise RuntimeError("No usable train/validation fields")
    if df.field_id.duplicated().any():
        raise RuntimeError("Duplicate field IDs in extracted dataset")
    print(f"Extracted {sum(df.split == 'train')} training fields and "
          f"{sum(df.split == 'val')} validation fields")
    return df


def train_alphaearth_field_rf(*, project_dir, crop_fields, crop_mapping,
                               n_estimators=300, min_samples_leaf=2,
                               min_field_pixels=10, random_state=42,
                               output_dir=None, tiles_dir=None,
                               reuse_features=True):
    """Train/evaluate RF on spatially separated fields; save model and CSVs."""
    root = Path(project_dir)
    output_dir = Path(output_dir or root / "models/alphaearth_field_random_forest")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "field_features.csv"
    if reuse_features and features_path.exists():
        df = pd.read_csv(features_path)
        print(f"Loaded cached features: {len(df)} fields")
    else:
        df = extract_field_features(
            project_dir=root, crop_fields=crop_fields, crop_mapping=crop_mapping,
            tiles_dir=tiles_dir, min_field_pixels=min_field_pixels)
        df.to_csv(features_path, index=False)
    columns = [f"A{i:02d}" for i in range(64)]
    train, val = df.loc[df.split == "train"].copy(), df.loc[df.split == "val"].copy()
    if not len(train) or not len(val) or set(train.field_id) & set(val.field_id):
        raise RuntimeError("Empty split or field-level leakage")
    if set(train.true_crop_id) - set(range(1, 31)):
        raise ValueError("Invalid crop IDs")
    rf = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample", n_jobs=-1,
        random_state=random_state)
    rf.fit(train[columns], train.true_crop_id)
    val["predicted_crop_id"] = rf.predict(val[columns]).astype(int)
    val["correct"] = val.true_crop_id == val.predicted_crop_id
    val["predicted_crop"] = val.predicted_crop_id.map({v:k for k,v in crop_mapping.items()})
    val["max_probability"] = rf.predict_proba(val[columns]).max(axis=1)
    cm = confusion_matrix(val.true_crop_id, val.predicted_crop_id, labels=np.arange(1, 31))
    hits, actual, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    union = actual + predicted - hits
    iou = np.divide(hits, union, out=np.full(30, np.nan), where=union>0)
    recall = np.divide(hits, actual, out=np.full(30, np.nan), where=actual>0)
    precision = np.divide(hits, predicted, out=np.full(30, np.nan), where=predicted>0)
    f1 = np.divide(2*hits, actual+predicted, out=np.full(30, np.nan), where=(actual+predicted)>0)
    inv = {v:k for k,v in crop_mapping.items()}
    by_crop = pd.DataFrame({
        "crop_id": np.arange(1,31), "crop": [inv[i] for i in range(1,31)],
        "true_fields": actual, "predicted_fields": predicted,
        "correct_fields": hits, "field_IoU": iou, "field_precision": precision,
        "field_recall": recall, "field_F1": f1})
    summary = pd.DataFrame([dict(model="alphaearth_rf", evaluated_fields=len(val),
        field_accuracy=accuracy_score(val.true_crop_id, val.predicted_crop_id),
        macro_field_IoU=float(np.nanmean(iou)),
        macro_field_F1=float(np.nanmean(f1)))])
    joblib.dump(rf, output_dir / "random_forest.joblib")
    by_crop.to_csv(output_dir / "field_metrics_by_crop.csv", index=False)
    summary.to_csv(output_dir / "field_summary.csv", index=False)
    val[["field_id", "area_ids", "crop", "true_crop_id", "predicted_crop", "predicted_crop_id",
         "correct", "pixels", "max_probability"]].to_csv(output_dir / "validation_field_predictions.csv", index=False)
    np.save(output_dir / "field_confusion_matrix.npy", cm)
    print("\nALPHAEARTH FIELD-LEVEL RANDOM FOREST")
    print(summary.to_string(index=False))
    previous = root / "data/processed/field_level_comparison/field_predictions_both_models.csv"
    if previous.is_file():
        old = pd.read_csv(previous)
        old = old.loc[old.model == "baseline", ["field_id", "correct"]]
        old_ids = set(old.field_id.astype(int))
        new_ids = set(val.field_id.astype(int))
        print(f"Compared with prior U-Net evaluation: {len(old_ids & new_ids)} common fields "
              f"(RF {len(new_ids)}, baseline {len(old_ids)})")
        if old_ids != new_ids:
            print("WARNING: field sets differ; do not directly compare aggregate scores.")
        else:
            print("Same  field IDs as U-Net baseline: direct field metric comparison is valid.")
    print(f"Saved results to: {output_dir}")
    return rf, summary, by_crop, val
