"""Fill missing months in multitemporal Sentinel-2 composites using clear-count rasters.

The source composites and clear-count rasters are never modified.

For each pixel and spectral band:
* existing monthly observations (clear count > 0) are preserved exactly;
* gaps between available months are linearly interpolated in time;
* gaps before the first or after the last observation use the nearest observed month;
* pixels without observations in any month remain 0 (NoData).

Use clear observation counts, not reflectance == 0, to identify missing values.
Outputs an 80-band (8 months x 10 bands) uint16 raster, an 8-band
fill-status raster, and a CSV of quality-control statistics.
"""

from __future__ import annotations

from pathlib import Path
import csv
import os
import uuid

import numpy as np
import rasterio


STATUS_LABELS = {0: "observed", 1: "interpolated", 2: "edge_filled", 3: "unfilled"}


def interpolate_monthly_composites(
    composites_dir: str | Path,
    clear_counts_dir: str | Path,
    output_dir: str | Path,
    *,
    bands_per_month: int = 10,
    overwrite: bool = False,
    area_ids: list[int] | None = None,
) -> list[dict]:
    """Interpolate missing monthly Sentinel-2 values for all composites in a folder.

    Parameters
    ----------
    composites_dir : path
        Folder containing germany_2023_area_<id>.tif (80-band images).
    clear_counts_dir : path
        Folder containing germany_2023_area_<id>_clear_counts.tif (8 bands).
    output_dir : path
        Separate destination; writes images/, fill_status/, and coverage_summary.csv.
    bands_per_month : int
        Spectral bands in each month's part of the stack (default 10).
    overwrite : bool
        If False, refuse to replace output images. Useful for restarting a run.
    area_ids : list[int] | None
        Optional subset of numeric area IDs; None processes every area.

    Returns
    -------
    list[dict]
        One per month and area, with coverage and fill percentages.
    """
    composites_dir = Path(composites_dir).resolve()
    clear_counts_dir = Path(clear_counts_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir in (composites_dir, clear_counts_dir):
        raise ValueError("Output directory must differ from both source directories")
    if bands_per_month < 1:
        raise ValueError("bands_per_month must be positive")
    images = sorted(composites_dir.glob("germany_2023_area_*.tif"))
    if area_ids is not None:
        allowed = {int(area_id) for area_id in area_ids}
        images = [p for p in images if int(p.stem.rsplit("_", 1)[-1]) in allowed]
    if not images:
        raise FileNotFoundError(f"No matching source composites found in {composites_dir}")

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "fill_status").mkdir(parents=True, exist_ok=True)
    rows = []

    for image_path in images:
        count_path = clear_counts_dir / f"{image_path.stem}_clear_counts.tif"
        if not count_path.is_file():
            raise FileNotFoundError(f"Missing clear-count raster: {count_path}")
        output_image = output_dir / "images" / image_path.name
        output_status = output_dir / "fill_status" / f"{image_path.stem}_fill_status.tif"
        if output_image.exists() and output_status.exists() and not overwrite:
            print(f"{image_path.stem}: outputs already exist; skipping")
            continue
        if (output_image.exists() or output_status.exists()) and not overwrite:
            raise FileExistsError(f"Partial outputs exist for {image_path.stem}. Check or use overwrite=True.")

        with rasterio.open(image_path) as src, rasterio.open(count_path) as cnt:
            if src.count % bands_per_month:
                raise ValueError(f"{image_path.name}: {src.count} not divisible by {bands_per_month}")
            nmonths = src.count // bands_per_month
            if nmonths != cnt.count:
                raise ValueError(f"{image_path.name}: {nmonths} months but {cnt.count} count bands")
            if (src.crs != cnt.crs or not src.transform.almost_equals(cnt.transform)
                    or src.width != cnt.width or src.height != cnt.height):
                raise ValueError(f"Raster and clear-count grid mismatch: {image_path.name}")
            if src.dtypes[0] != "uint16":
                raise ValueError(f"Expected uint16 reflectance input: {image_path.name}")
            data = src.read().reshape(nmonths, bands_per_month, src.height, src.width)
            counts = cnt.read()
            image_profile = src.profile.copy()
            counts_profile = cnt.profile.copy()
            descriptions = src.descriptions
            status_descriptions = cnt.descriptions
            original_tags = src.tags()

        valid = counts > 0
        t = np.arange(nmonths, dtype=np.int16)[:, None, None]
        left = np.maximum.accumulate(np.where(valid, t, -1), axis=0)
        right = np.minimum.accumulate(np.where(valid, t, nmonths)[::-1], axis=0)[::-1]
        never_observed = ~valid.any(axis=0)

        # 0 observed; 1 interior interpolation; 2 nearest-month edge fill;
        # 3 no usable observations during the entire growing season.
        status = np.zeros_like(counts, dtype=np.uint8)
        for month in range(nmonths):
            missing = ~valid[month]
            inside = missing & (left[month] >= 0) & (right[month] < nmonths)
            edge = missing & ~inside & ~never_observed
            status[month, inside] = 1
            status[month, edge] = 2
            status[month, never_observed] = 3

            rows.append({
                "image": image_path.name,
                "month_index": month + 1,
                "observed_pct": round(100 * np.mean(status[month] == 0), 2),
                "interpolated_pct": round(100 * np.mean(status[month] == 1), 2),
                "edge_filled_pct": round(100 * np.mean(status[month] == 2), 2),
                "unfilled_pct": round(100 * np.mean(status[month] == 3), 2),
            })

        # Interpolate one 2-D band at a time to avoid a large temporary array.
        result = data.copy()
        for b in range(bands_per_month):
            series = data[:, b]
            for month in range(nmonths):
                missing = ~valid[month] & ~never_observed
                if not missing.any():
                    continue
                ys, xs = np.nonzero(missing)
                lidx = np.where(left[month, ys, xs] >= 0,
                                left[month, ys, xs], right[month, ys, xs])
                ridx = np.where(right[month, ys, xs] < nmonths,
                                right[month, ys, xs], left[month, ys, xs])
                lv = series[lidx, ys, xs].astype(np.float32)
                rv = series[ridx, ys, xs].astype(np.float32)
                spacing = ridx - lidx
                fraction = np.divide(month - lidx, spacing,
                                     out=np.zeros_like(lv), where=spacing != 0)
                values = lv + fraction * (rv - lv)
                result[month, b, ys, xs] = np.rint(values).clip(0, 65535).astype(np.uint16)

        # Keep entire time series missing wherever no month has a clear observation.
        result[:, :, never_observed] = 0

        image_profile.update(count=nmonths * bands_per_month, dtype="uint16", nodata=0)
        counts_profile.update(count=nmonths, dtype="uint8", nodata=None)
        temp_image = output_image.with_name(output_image.stem + "." + uuid.uuid4().hex + ".tmp.tif")
        temp_status = output_status.with_name(output_status.stem + "." + uuid.uuid4().hex + ".tmp.tif")
        try:
            with rasterio.open(temp_image, "w", **image_profile) as dst:
                dst.write(result.reshape(nmonths * bands_per_month, *result.shape[-2:]))
                for idx, name in enumerate(descriptions, start=1):
                    if name:
                        dst.set_band_description(idx, name)
                dst.update_tags(**original_tags)
                dst.update_tags(temporal_gap_filling="linear_inside_nearest_outside",
                                gap_filling_source="clear_observation_counts")
            with rasterio.open(temp_status, "w", **counts_profile) as dst:
                dst.write(status)
                for idx in range(nmonths):
                    dst.set_band_description(idx + 1, f"{status_descriptions[idx] or f'month_{idx + 1}'}_fill_status")
                dst.update_tags(status_codes="0=observed;1=interpolated;2=edge_filled;3=unfilled")
            os.replace(temp_image, output_image)
            os.replace(temp_status, output_status)
        finally:
            temp_image.unlink(missing_ok=True)
            temp_status.unlink(missing_ok=True)

        print(f"{image_path.stem}: completed; "
              f"pixels missing in every month: {never_observed.mean():.2%}")

    if rows:
        csv_path = output_dir / "coverage_summary.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
