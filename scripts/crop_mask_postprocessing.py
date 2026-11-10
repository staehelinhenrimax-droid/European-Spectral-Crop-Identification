"""Validate multitemporal Sentinel-2 rasters and create unambiguous crop masks.

Keeps partially intersecting crop fields; ignores pixels covered by multiple
field polygons and, optionally, within a buffer of field boundaries.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from shapely.geometry import box


def create_checked_crop_masks(
    selected_areas,
    crop_fields,
    crop_mapping,
    images_dir,
    masks_dir,
    boundary_buffer_m=5.0,
    expected_bands=80,
    bounds_tolerance_pixels=1.0,
):
    """Create aligned masks (crop IDs 1..N, ignore=255) for each image.

    Areas and fields can be in any defined CRS. Raster CRS must be projected
    in metres. A field need only intersect the raster to contribute labels.
    Pixels hit by two or more field polygons, or near a field boundary, are
    ignored. All input images and vector objects are unchanged.
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if not (0 <= boundary_buffer_m):
        raise ValueError('boundary_buffer_m must be >= 0')
    if crop_fields.crs is None or selected_areas.crs is None:
        raise ValueError('Both input GeoDataFrames require a CRS')
    if 'area_id' not in selected_areas or 'hcat4_name' not in crop_fields:
        raise ValueError('Expected area_id and hcat4_name columns')
    if selected_areas.area_id.duplicated().any():
        raise ValueError('Duplicate area_id values')
    if not crop_mapping or len(crop_mapping) > 254 or any(
        not isinstance(v, (int, np.integer)) or not 1 <= v <= 254
        for v in crop_mapping.values()
    ) or len(set(crop_mapping.values())) != len(crop_mapping):
        raise ValueError('crop_mapping requires unique integer IDs from 1 to 254')

    projected_fields_cache = {}
    records = []
    for _, area in selected_areas.iterrows():
        area_id = int(area['area_id'])
        image_path = images_dir / f'germany_2023_area_{area_id}.tif'
        mask_path = masks_dir / f'germany_mask_{area_id}.tif'
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with rasterio.open(image_path) as src:
            if src.crs is None or not src.crs.is_projected:
                raise ValueError(f'Area {area_id}: raster must use a projected CRS')
            # UTM or other metre-based projection, not degrees/feet.
            units = [str(a).lower() for a in (src.crs.linear_units or '',)]
            if not any('met' in u for u in units):
                raise ValueError(f'Area {area_id}: raster CRS units are not metres')
            if expected_bands is not None and src.count != expected_bands:
                raise ValueError(f'Area {area_id}: {src.count} bands, expected {expected_bands}')
            if src.res[0] <= 0 or src.res[1] <= 0:
                raise ValueError(f'Area {area_id}: invalid raster resolution')
            crs, bounds, transform = src.crs, src.bounds, src.transform
            height, width = src.height, src.width
            profile = src.profile.copy()

        # Check raster footprint agrees with selected grid cell (allow <=1 pixel).
        area_projected = gpd.GeoSeries([area.geometry], crs=selected_areas.crs).to_crs(crs).iloc[0]
        expected = area_projected.bounds
        differences = [abs(a-b) for a,b in zip(bounds, expected)]
        tol = bounds_tolerance_pixels * max(abs(transform.a), abs(transform.e))
        if max(differences) > tol:
            raise ValueError(
                f'Area {area_id}: raster bounds {tuple(bounds)} differ from '
                f'selected area bounds {expected} by up to {max(differences):.1f} m '
                f'(allowed {tol:.1f} m)'
            )

        key = crs.to_string()
        if key not in projected_fields_cache:
            projected_fields_cache[key] = crop_fields.to_crs(crs)
        all_fields = projected_fields_cache[key]
        footprint = box(*bounds)

        # Select fields that INTERSECT the footprint, not only fully contained fields.
        candidates = all_fields.loc[all_fields.geometry.intersects(footprint)].copy()
        candidates = candidates.loc[
            candidates['hcat4_name'].isin(crop_mapping)
            & candidates.geometry.notna()
            & ~candidates.geometry.is_empty
        ].copy()
        # Keep original outlines for buffering: the raster's outer boundary is
        # not a crop-field boundary and should not erase partial-field pixels.
        original_boundaries = {idx: geom.boundary for idx, geom in candidates.geometry.items()}
        candidates['geometry'] = candidates.geometry.intersection(footprint)
        candidates = candidates.loc[
            ~candidates.geometry.is_empty & (candidates.geometry.area > 0)
        ].copy()
        if candidates.empty:
            raise ValueError(f'Area {area_id}: no mapped crop fields overlap raster')

        shapes = [
            (geom, int(crop_mapping[name]))
            for name, geom in zip(candidates['hcat4_name'], candidates.geometry)
        ]
        shape = (height, width)
        # Last-wins class raster is safe only because overlapping pixels are erased.
        mask = rasterize(shapes, out_shape=shape, transform=transform,
                         fill=255, dtype='uint8', all_touched=False)
        overlap_count = rasterize(((geom, 1) for geom, _ in shapes),
                                  out_shape=shape, transform=transform,
                                  fill=0, dtype='uint16', all_touched=False,
                                  merge_alg=MergeAlg.add)
        ambiguous = overlap_count >= 2
        mask[ambiguous] = 255

        boundary_pixels = np.zeros(shape, dtype=bool)
        if boundary_buffer_m > 0:
            zones = [original_boundaries[idx].buffer(boundary_buffer_m)
                     for idx in candidates.index]
            boundary_pixels = rasterize(((g, 1) for g in zones),
                                        out_shape=shape, transform=transform,
                                        fill=0, dtype='uint8', all_touched=False).astype(bool)
            mask[boundary_pixels] = 255

        profile.update(driver='GTiff', count=1, dtype='uint8', nodata=255,
                       compress='deflate')
        # Avoid confusing old masks with the new dataset.
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask, 1)
        with rasterio.open(mask_path) as check:
            assert check.crs == crs and check.transform == transform
            assert check.width == width and check.height == height
            assert set(map(int, np.unique(check.read(1)))) <= set(crop_mapping.values()) | {255}

        records.append({
            'area_id': area_id,
            'crs': str(crs),
            'bounds_match': True,
            'raster_shape': f'{height}x{width}',
            'fields_intersecting': len(candidates),
            'overlap_pixels': int(ambiguous.sum()),
            'boundary_pixels': int(boundary_pixels.sum()),
            'labeled_pixels': int(np.sum(mask != 255)),
            'labeled_percent': round(float(np.mean(mask != 255) * 100), 2),
        })
    return pd.DataFrame(records)
