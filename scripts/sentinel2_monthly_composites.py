"""Sentinel-2 L2A monthly cloud-masked median composites with OmniCloudMask.

Produces 80-channel image (8 months x 10 bands), plus an 8-channel
clear-observation-count raster and JSON provenance for each area.

Dependencies: geopandas, numpy, pandas, planetary-computer, pystac-client,
             rasterio, omnicloudmask (and its PyTorch dependencies).
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from pystac_client import Client
import planetary_computer as pc

BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12")
BANDS_20M = {"B05", "B06", "B07", "B8A", "B11", "B12"}


def _read_band(asset_url, profile, band):
    """Read a Sentinel-2 asset onto the fixed AOI grid; 0 is nodata."""
    with rasterio.open(asset_url) as src:
        with WarpedVRT(
            src,
            crs=profile["crs"],
            transform=profile["transform"],
            width=profile["width"],
            height=profile["height"],
            src_nodata=src.nodata if src.nodata is not None else 0,
            nodata=0,
            resampling=(Resampling.bilinear if band in BANDS_20M else Resampling.nearest),
        ) as vrt:
            return vrt.read(1).astype(np.uint16, copy=False)


def _scene_items(catalog, geometry_wgs84, year, month, max_cloud_cover, max_scenes):
    start = f"{year}-{month:02d}-01"
    if month == 12:
        end = f"{year+1}-01-01"
    else:
        end = f"{year}-{month+1:02d}-01"
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry_wgs84.__geo_interface__,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lte": max_cloud_cover}},
    )
    items = [item for item in search.items() if item.datetime is not None and item.datetime.year == year and item.datetime.month == month]
    items.sort(key=lambda item: (float(item.properties.get("eo:cloud_cover", 100) if item.properties.get("eo:cloud_cover") is not None else 100), item.datetime))
    # Avoid duplicate acquisitions from overlapping S2 tiles on the same day.
    chosen = []
    seen_dates = set()
    for item in items:
        date = item.datetime.date()
        if date in seen_dates:
            continue
        if not all(band in item.assets for band in BANDS):
            continue
        chosen.append(item)
        seen_dates.add(date)
        if len(chosen) == max_scenes:
            break
    return chosen


def _grid_from_geometry(geom, crs, pixel_size):
    minx, miny, maxx, maxy = geom.bounds
    width_f, height_f = (maxx - minx) / pixel_size, (maxy - miny) / pixel_size
    width, height = round(width_f), round(height_f)
    if width < 1 or height < 1 or not np.isclose(width_f, width, atol=1e-5) or not np.isclose(height_f, height, atol=1e-5):
        raise ValueError("Area dimensions must be multiples of pixel_size; try an area_size divisible by 10 m.")
    if not np.isclose(geom.area, geom.envelope.area, rtol=1e-5):
        raise ValueError("Expected full rectangular grid-cell polygons as selected_areas geometry.")
    return {"driver": "GTiff", "width": width, "height": height,
            "crs": crs, "transform": from_bounds(minx, miny, maxx, maxy, width, height)}


def create_monthly_composites(
    selected_areas,
    output_dir,
    year=2023,
    months=range(3, 11),
    scenes_per_month=5,
    max_cloud_cover=60,
    pixel_size=10,
    inference_device=None,
    overwrite=False,
):
    """Build 10-band monthly medians and stack them for every study area.

    Parameters
    ----------
    selected_areas : geopandas.GeoDataFrame
        Has an integer 'area_id' and square geometry in a metric projected CRS.
    output_dir : str or Path
        Output: images/germany_<year>_area_<id>.tif (80 bands),
        clear_counts/...tif (8 bands), metadata/...json per area.
    months : iterable of int
        Default March to October. An output raster has len(months)*10 bands.
    scenes_per_month : int
        Maximum *different acquisition dates* used for each monthly median.
    max_cloud_cover : float
        STAC scene-level eo:cloud_cover maximum, not AOI-level percentage.
    inference_device : str or None
        'mps', 'cpu', or 'cuda'; None lets OmniCloudMask choose.

    Missing clear observations are stored as zero (raster nodata), not as valid
    reflectance. Inspect clear_counts before training; do not silently treat
    missing months as cloud-free data. No interpolation is performed.
    """
    if selected_areas.crs is None or not selected_areas.crs.is_projected:
        raise ValueError("selected_areas must have a projected metric CRS (e.g., EPSG:32632).")
    if "area_id" not in selected_areas:
        raise KeyError("selected_areas requires an 'area_id' column.")
    if selected_areas["area_id"].duplicated().any():
        raise ValueError("area_id values must be unique.")
    if scenes_per_month < 1 or not 0 <= max_cloud_cover <= 100:
        raise ValueError("scenes_per_month >= 1 and max_cloud_cover between 0 and 100 are required.")
    months = tuple(months)
    if not months or len(set(months)) != len(months) or any(not 1 <= m <= 12 for m in months):
        raise ValueError("months must be unique calendar months between 1 and 12.")

    # Import lazily: notebook startup need not initialize PyTorch for other cells.
    from omnicloudmask import predict_from_array

    output_dir = Path(output_dir)
    for folder in ("images", "clear_counts", "metadata"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    wgs84 = selected_areas.to_crs("EPSG:4326").set_index("area_id")
    results = []

    for _, area in selected_areas.iterrows():
        area_id = int(area["area_id"])
        stem = f"germany_{year}_area_{area_id}"
        image_path = output_dir / "images" / f"{stem}.tif"
        count_path = output_dir / "clear_counts" / f"{stem}_clear_counts.tif"
        meta_path = output_dir / "metadata" / f"{stem}.json"
        if not overwrite and (image_path.exists() or count_path.exists() or meta_path.exists()):
            if all(path.exists() for path in (image_path, count_path, meta_path)):
                with rasterio.open(image_path) as check:
                    if check.count == len(months)*len(BANDS) and all(check.descriptions):
                        print(f"Area {area_id}: reusing completed stack")
                        results.append({"area_id": area_id, "image": str(image_path), "status": "existing"})
                        continue
            raise ValueError(f"Area {area_id}: incomplete/old output exists. Inspect or call overwrite=True: {image_path}")

        profile = _grid_from_geometry(area.geometry, selected_areas.crs, pixel_size)
        h, w = profile["height"], profile["width"]
        monthly_bands = []
        monthly_counts = []
        records = []
        for month in months:
            items = _scene_items(catalog, wgs84.loc[area_id].geometry, year, month, max_cloud_cover, scenes_per_month)
            scenes = []
            n_clear = np.zeros((h, w), dtype=np.uint8)
            item_records = []
            for item in items:
                try:
                    signed = pc.sign(item)  # Sign immediately before reading to limit expiry.
                    band_arrays = [_read_band(signed.assets[b].href, profile, b) for b in BANDS]
                    data = np.stack(band_arrays)
                    # Require every channel to have valid data at a pixel.
                    valid = np.all(data != 0, axis=0)
                    cloud_input = data[[2, 1, 3], :, :].astype(np.float32) / 10000.0  # Red, Green, NIR
                    # Restore nodata to zero in OCM input and exclude it after inference.
                    cloud_input[:, ~valid] = 0.0
                    ocm = predict_from_array(cloud_input, inference_device=inference_device)
                    cloud_classes = np.asarray(ocm).squeeze()
                    if cloud_classes.shape != (h, w):
                        raise ValueError(f"OmniCloudMask returned unexpected shape {cloud_classes.shape}")
                    clear = valid & (cloud_classes == 0)
                    n_clear += clear.astype(np.uint8)
                    scenes.append(np.where(clear[None, :, :], data, np.nan).astype(np.float32))
                    item_records.append({
                        "stac_item_id": item.id,
                        "datetime": item.datetime.isoformat(),
                        "scene_cloud_cover_pct": item.properties.get("eo:cloud_cover"),
                        "aoi_clear_fraction": round(float(clear.mean()), 4),
                    })
                except Exception as exc:
                    warnings.warn(f"Area {area_id} month {month} scene {item.id} failed: {exc}")
            if scenes:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
                    median = np.nanmedian(np.stack(scenes, axis=0), axis=0)
                monthly = np.nan_to_num(median, nan=0.0).clip(0, 65535).astype(np.uint16)
            else:
                monthly = np.zeros((len(BANDS), h, w), dtype=np.uint16)
            monthly_bands.append(monthly)
            monthly_counts.append(n_clear)
            pct = 100 * np.mean(n_clear > 0)
            print(f"Area {area_id}, {year}-{month:02d}: {len(scenes)}/{len(items)} scenes, {pct:.1f}% pixels with clear observation")
            records.append({"month": month, "candidate_scenes": len(items), "usable_scenes": len(scenes),
                            "monthly_coverage_pct": round(float(pct), 2), "scenes": item_records})

        out_profile = dict(profile, count=len(months)*len(BANDS), dtype="uint16", nodata=0, compress="deflate")
        count_profile = dict(profile, count=len(months), dtype="uint8", nodata=None, compress="deflate")
        temp_img = image_path.with_suffix(".partial.tif")
        temp_cnt = count_path.with_suffix(".partial.tif")
        try:
            with rasterio.open(temp_img, "w", **out_profile) as dst:
                for midx, month in enumerate(months):
                    dst.write(monthly_bands[midx], indexes=list(range(midx*10+1, midx*10+11)))
                    for bidx, band in enumerate(BANDS):
                        dst.set_band_description(midx*10+bidx+1, f"{year}-{month:02d}_{band}")
                dst.update_tags(description="Month-wise median of OmniCloudMask clear pixels", year=str(year),
                                months=",".join(map(str, months)), scene_cloud_threshold=str(max_cloud_cover))
            with rasterio.open(temp_cnt, "w", **count_profile) as dst:
                for midx, month in enumerate(months):
                    dst.write(monthly_counts[midx], midx+1)
                    dst.set_band_description(midx+1, f"{year}-{month:02d}_clear_observation_count")
            temp_img.replace(image_path)
            temp_cnt.replace(count_path)
            meta_path.write_text(json.dumps({"area_id": area_id, "year": year, "months": list(months),
                "bands_per_month": list(BANDS), "max_scene_cloud_cover": max_cloud_cover,
                "maximum_scenes_per_month": scenes_per_month, "monthly_details": records}, indent=2))
        finally:
            temp_img.unlink(missing_ok=True)
            temp_cnt.unlink(missing_ok=True)
        results.append({"area_id": area_id, "image": str(image_path), "status": "created",
                        "min_month_coverage_pct": min(r["monthly_coverage_pct"] for r in records)})
    return pd.DataFrame(results)
