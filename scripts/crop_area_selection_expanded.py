"""Select geographically separated crop-classification areas from field polygons.

All size/distance arguments are in metres. Crop counts are based on complete
fields contained within each candidate square, not centroid intersections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


def select_crop_areas(
    crop_fields: gpd.GeoDataFrame,
    area_size: float = 5120,
    n_areas: int = 50,
    min_distance: float = 10_000,
    min_crop_types: int = 10,
    crop_column: str = "hcat4_name",
    target_crs: str = "EPSG:32632",
    first_area_id: int = 1,
) -> gpd.GeoDataFrame:
    """Select up to ``n_areas`` square areas with diverse, complete crop fields.

    Candidates are grid cells containing at least ``min_crop_types`` complete
    fields of distinct types. Each iteration scores a candidate by the weighted
    increase in square-root field counts over previously selected areas.
    Rare crop types receive greater weight, calculated from field frequencies
    in the supplied dataset. Selected squares remain at least ``min_distance``
    apart, measured boundary-to-boundary.

    Returns a GeoDataFrame with ``area_id``, ``grid_id``, ``n_crop_types``,
    ``n_fields``, ``score``, and ``geometry`` in ``target_crs``. The score is
    the marginal score *at selection time*, not an absolute quality metric.

    Assumes each row represents one field. Fields crossing the square boundary
    do not contribute to the crop counts or scores.
    """
    if crop_fields.empty:
        raise ValueError("crop_fields is empty")
    if crop_fields.crs is None:
        raise ValueError("crop_fields must have a CRS")
    if crop_column not in crop_fields.columns:
        raise KeyError(f"Missing crop column: {crop_column}")
    if area_size <= 0 or min_distance < 0 or n_areas < 1 or min_crop_types < 1:
        raise ValueError("Require area_size>0, min_distance>=0, n_areas>=1, min_crop_types>=1")

    fields = crop_fields[[crop_column, "geometry"]].dropna(subset=[crop_column, "geometry"]).to_crs(target_crs)
    fields = fields.loc[~fields.geometry.is_empty].reset_index(drop=True)
    if fields.empty:
        raise ValueError("No valid crop fields remain after removing missing values")
    if not fields.crs.is_projected:
        raise ValueError("target_crs must be a projected CRS with metre units")

    minx, miny, maxx, maxy = fields.total_bounds
    xs = np.arange(np.floor(minx / area_size) * area_size, maxx, area_size)
    ys = np.arange(np.floor(miny / area_size) * area_size, maxy, area_size)
    grid = gpd.GeoDataFrame(
        {"grid_id": np.arange(len(xs) * len(ys), dtype=int)},
        geometry=[box(x, y, x + area_size, y + area_size) for x in xs for y in ys],
        crs=fields.crs,
    )

    # Screening by centroid avoids expensive full-polygon tests for empty cells.
    centroids = gpd.GeoDataFrame(
        {crop_column: fields[crop_column]}, geometry=fields.geometry.centroid,
        crs=fields.crs,
    )
    centroid_hits = gpd.sjoin(centroids, grid, how="inner", predicate="within")
    centroid_diversity = centroid_hits.groupby("grid_id")[crop_column].nunique()
    grid = grid[grid["grid_id"].isin(centroid_diversity[centroid_diversity >= min_crop_types].index)].copy()
    if grid.empty:
        raise ValueError("No candidates meet the centroid diversity threshold")

    # Only entire field geometries fully inside each grid cell count.
    complete = gpd.sjoin(fields, grid, how="inner", predicate="within")
    per_crop = complete.groupby(["grid_id", crop_column]).size().unstack(fill_value=0)
    per_crop = per_crop.reindex(grid["grid_id"], fill_value=0)
    eligible = per_crop.gt(0).sum(axis=1) >= min_crop_types
    per_crop = per_crop.loc[eligible]
    grid = grid[grid["grid_id"].isin(per_crop.index)].copy()
    if grid.empty:
        raise ValueError("No candidates contain enough distinct *complete* crop fields")

    # Global rarity weights, and a vectorized marginal-gain calculation.
    global_counts = fields[crop_column].value_counts()
    priorities = 1 / np.sqrt(global_counts.to_numpy(dtype=float) + 1)
    priorities /= priorities.max()
    per_crop = per_crop.reindex(columns=global_counts.index, fill_value=0)
    counts_matrix = per_crop.to_numpy(dtype=float)
    ids = per_crop.index.to_numpy()
    weights = priorities
    selected_counts = np.zeros(len(global_counts), dtype=float)

    geometry_by_id = grid.set_index("grid_id").geometry
    allowed = np.ones(len(ids), dtype=bool)
    choices = []

    for _ in range(n_areas):
        available = np.flatnonzero(allowed)
        if not len(available):
            break
        additions = counts_matrix[available]
        gains = np.sqrt(selected_counts[None, :] + additions) - np.sqrt(selected_counts[None, :])
        scores = gains @ weights
        pick = int(available[np.argmax(scores)])
        area_id = ids[pick]
        geom = geometry_by_id.loc[area_id]
        choices.append({
            "grid_id": int(area_id),
            "n_crop_types": int(np.count_nonzero(counts_matrix[pick])),
            "n_fields": int(counts_matrix[pick].sum()),
            "score": float(np.max(scores)),
            "geometry": geom,
        })
        selected_counts += counts_matrix[pick]

        # Intersecting the buffered selected square violates min_distance.
        exclusion = geom.buffer(min_distance)
        nearby = grid.loc[grid.geometry.intersects(exclusion), "grid_id"]
        allowed &= ~np.isin(ids, nearby.to_numpy())

    selected = gpd.GeoDataFrame(choices, crs=grid.crs)
    selected.insert(0, "area_id", np.arange(first_area_id, first_area_id + len(selected)))
    return selected


def plot_crop_areas(areas: gpd.GeoDataFrame, figsize=(11, 9), ax=None):
    """Plot the selected area boundaries, returning (figure, axes)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    areas.boundary.plot(ax=ax, color="blue", linewidth=1.5)
    ax.set_title(f"Selected crop areas ({len(areas)})")
    ax.set_aspect("equal")
    return fig, ax


def select_additional_crop_areas(
    crop_fields: gpd.GeoDataFrame,
    existing_areas: gpd.GeoDataFrame,
    area_size: float = 5120,
    n_additional: int = 100,
    min_distance: float = 10_000,
    min_crop_types: int = 10,
    crop_column: str = "hcat4_name",
    target_crs: str = "EPSG:32632",
    first_area_id: int | None = None,
) -> gpd.GeoDataFrame:
    """Select up to ``n_additional`` new areas while keeping existing ones fixed.

    Candidates are grid cells containing at least ``min_crop_types`` complete
    fields of distinct types. Each iteration scores a candidate by the weighted
    increase in square-root field counts over previously selected areas.
    Rare crop types receive greater weight, calculated from field frequencies
    in the supplied dataset. Selected squares remain at least ``min_distance``
    apart, measured boundary-to-boundary.

    Returns a GeoDataFrame with ``area_id``, ``grid_id``, ``n_crop_types``,
    ``n_fields``, ``score``, and ``geometry`` in ``target_crs``. The score is
    the marginal score *at selection time*, not an absolute quality metric.

    Assumes each row represents one field. Fields crossing the square boundary
    do not contribute to the crop counts or scores.
    """
    if existing_areas.empty or existing_areas.crs is None or 'area_id' not in existing_areas:
        raise ValueError('existing_areas needs existing area IDs, geometries and CRS')
    if existing_areas['area_id'].duplicated().any():
        raise ValueError('existing_areas contains duplicate IDs')
    if first_area_id is None:
        first_area_id = int(existing_areas['area_id'].max()) + 1
    if first_area_id <= int(existing_areas['area_id'].max()):
        raise ValueError('New area IDs must not overlap existing area IDs')
    n_areas = n_additional
    if crop_fields.empty:
        raise ValueError("crop_fields is empty")
    if crop_fields.crs is None:
        raise ValueError("crop_fields must have a CRS")
    if crop_column not in crop_fields.columns:
        raise KeyError(f"Missing crop column: {crop_column}")
    if area_size <= 0 or min_distance < 0 or n_areas < 1 or min_crop_types < 1:
        raise ValueError("Require area_size>0, min_distance>=0, n_areas>=1, min_crop_types>=1")

    fields = crop_fields[[crop_column, "geometry"]].dropna(subset=[crop_column, "geometry"]).to_crs(target_crs)
    fields = fields.loc[~fields.geometry.is_empty].reset_index(drop=True)
    if fields.empty:
        raise ValueError("No valid crop fields remain after removing missing values")
    if not fields.crs.is_projected:
        raise ValueError("target_crs must be a projected CRS with metre units")

    minx, miny, maxx, maxy = fields.total_bounds
    xs = np.arange(np.floor(minx / area_size) * area_size, maxx, area_size)
    ys = np.arange(np.floor(miny / area_size) * area_size, maxy, area_size)
    grid = gpd.GeoDataFrame(
        {"grid_id": np.arange(len(xs) * len(ys), dtype=int)},
        geometry=[box(x, y, x + area_size, y + area_size) for x in xs for y in ys],
        crs=fields.crs,
    )

    # Screening by centroid avoids expensive full-polygon tests for empty cells.
    centroids = gpd.GeoDataFrame(
        {crop_column: fields[crop_column]}, geometry=fields.geometry.centroid,
        crs=fields.crs,
    )
    centroid_hits = gpd.sjoin(centroids, grid, how="inner", predicate="within")
    centroid_diversity = centroid_hits.groupby("grid_id")[crop_column].nunique()
    grid = grid[grid["grid_id"].isin(centroid_diversity[centroid_diversity >= min_crop_types].index)].copy()
    if grid.empty:
        raise ValueError("No candidates meet the centroid diversity threshold")

    # Only entire field geometries fully inside each grid cell count.
    complete = gpd.sjoin(fields, grid, how="inner", predicate="within")
    per_crop = complete.groupby(["grid_id", crop_column]).size().unstack(fill_value=0)
    per_crop = per_crop.reindex(grid["grid_id"], fill_value=0)
    eligible = per_crop.gt(0).sum(axis=1) >= min_crop_types
    per_crop = per_crop.loc[eligible]
    grid = grid[grid["grid_id"].isin(per_crop.index)].copy()
    if grid.empty:
        raise ValueError("No candidates contain enough distinct *complete* crop fields")

    # Global rarity weights, and a vectorized marginal-gain calculation.
    global_counts = fields[crop_column].value_counts()
    priorities = 1 / np.sqrt(global_counts.to_numpy(dtype=float) + 1)
    priorities /= priorities.max()
    per_crop = per_crop.reindex(columns=global_counts.index, fill_value=0)
    counts_matrix = per_crop.to_numpy(dtype=float)
    ids = per_crop.index.to_numpy()
    weights = priorities
    selected_counts = np.zeros(len(global_counts), dtype=float)

    geometry_by_id = grid.set_index("grid_id").geometry
    allowed = np.ones(len(ids), dtype=bool)
    choices = []

    # Reserve all existing locations and their boundary-to-boundary buffer.
    fixed = existing_areas[['area_id', 'geometry']].to_crs(target_crs).copy()
    if not fixed.geometry.is_valid.all():
        raise ValueError('Invalid existing-area geometries')
    if not np.allclose(fixed.geometry.area.to_numpy(), area_size ** 2, rtol=0, atol=1):
        raise ValueError('Existing polygons do not have the expected area_size')
    fixed_exclusion = fixed.geometry.buffer(min_distance).union_all()
    allowed &= ~grid.set_index('grid_id').loc[ids].geometry.intersects(fixed_exclusion).to_numpy()

    # Account for crops already represented in the original areas when ranking
    # new squares, rather than treating all 30 classes as unseen.
    existing_complete = gpd.sjoin(fields, fixed, how='inner', predicate='within')
    initial_counts = existing_complete[crop_column].value_counts()
    selected_counts += initial_counts.reindex(global_counts.index, fill_value=0).to_numpy(dtype=float)

    for _ in range(n_areas):
        available = np.flatnonzero(allowed)
        if not len(available):
            break
        additions = counts_matrix[available]
        gains = np.sqrt(selected_counts[None, :] + additions) - np.sqrt(selected_counts[None, :])
        scores = gains @ weights
        pick = int(available[np.argmax(scores)])
        area_id = ids[pick]
        geom = geometry_by_id.loc[area_id]
        choices.append({
            "grid_id": int(area_id),
            "n_crop_types": int(np.count_nonzero(counts_matrix[pick])),
            "n_fields": int(counts_matrix[pick].sum()),
            "score": float(np.max(scores)),
            "geometry": geom,
        })
        selected_counts += counts_matrix[pick]

        # Intersecting the buffered selected square violates min_distance.
        exclusion = geom.buffer(min_distance)
        nearby = grid.loc[grid.geometry.intersects(exclusion), "grid_id"]
        allowed &= ~np.isin(ids, nearby.to_numpy())

    if not choices:
        raise ValueError('No additional areas satisfy crop diversity and spacing constraints')
    selected = gpd.GeoDataFrame(choices, crs=grid.crs)
    selected.insert(0, "area_id", np.arange(first_area_id, first_area_id + len(selected)))
    return selected




# ---------------------------------------------------------------------------
# Standalone expanded-areas notebook helpers
# ---------------------------------------------------------------------------
from pathlib import Path
import re

import rasterio
from shapely.geometry import box


def load_existing_areas_from_rasters(
    project_dir: str | Path,
    reference_images_dir: str | Path | None = None,
    original_tiles_dir: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Recover the exact original area bounds and geographic splits from disk.

    Unlike reconstructing a new selection, this preserves the spatial locations
    of the original 50 areas. Each split is inferred from the original tile
    filenames; the imagery files supply the exact footprint and CRS.
    """
    project_dir = Path(project_dir)
    reference_images_dir = Path(reference_images_dir or (
        project_dir / "data/processed/sentinel2_monthly_interpolated/images"
    ))
    original_tiles_dir = Path(original_tiles_dir or (
        project_dir / "data/processed/tiles_80band"
    ))

    original_ids = {}
    pattern = re.compile(r"^germany_2023_area_(\d+)_(\d+)\.tif$")
    for split in ("train", "val", "test"):
        split_dir = original_tiles_dir / split / "images"
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing original tile directory: {split_dir}")
        found = 0
        for tile_path in split_dir.glob("*.tif"):
            match = pattern.fullmatch(tile_path.name)
            if match is None:
                continue
            area_id = int(match.group(1))
            prior = original_ids.setdefault(area_id, split)
            if prior != split:
                raise ValueError(f"Area {area_id} appears in both {prior} and {split}")
            found += 1
        if not found:
            raise ValueError(f"No original tiles found for {split}: {split_dir}")

    rows = []
    crs = None
    for area_id, split in sorted(original_ids.items()):
        raster_path = reference_images_dir / f"germany_2023_area_{area_id}.tif"
        if not raster_path.is_file():
            raise FileNotFoundError(f"Original area image missing: {raster_path}")
        with rasterio.open(raster_path) as raster:
            if raster.crs is None:
                raise ValueError(f"Missing CRS: {raster_path}")
            if crs is None:
                crs = raster.crs
            elif raster.crs != crs:
                raise ValueError("Original area images do not share a CRS")
            if raster.width != 512 or raster.height != 512:
                raise ValueError(f"Unexpected area size in pixels: {raster_path}")
            if abs(raster.res[0] - 10) > 1e-6 or abs(raster.res[1] - 10) > 1e-6:
                raise ValueError(f"Unexpected pixel resolution: {raster_path}")
            rows.append({
                "area_id": area_id,
                "split": split,
                "geometry": box(*raster.bounds),
            })

    if len(rows) != 50 or set(original_ids) != set(range(1, 51)):
        raise ValueError(
            f"Expected original area IDs 1–50, found {len(rows)} areas: "
            f"{sorted(original_ids)}"
        )
    areas = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if set(areas["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test in original tiles")
    print("Recovered exact original raster footprints:", len(areas))
    print(areas["split"].value_counts().to_string())
    return areas


def save_expanded_areas(
    project_dir: str | Path,
    original_areas: gpd.GeoDataFrame,
    new_areas: gpd.GeoDataFrame,
    destination_dir: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Save separate new/all-areas GeoParquets without modifying originals."""
    project_dir = Path(project_dir)
    destination_dir = Path(destination_dir or (
        project_dir / "data/processed/area_selection_expanded"
    ))
    original_areas = original_areas.to_crs(new_areas.crs).copy()
    if "split" not in original_areas:
        raise ValueError("Original areas must contain a split column")
    if set(new_areas.area_id) & set(original_areas.area_id):
        raise ValueError("New and original area IDs overlap")
    new_areas = new_areas.copy()
    new_areas["split"] = "train"
    expanded = gpd.GeoDataFrame(
        pd.concat([original_areas, new_areas], ignore_index=True),
        geometry="geometry",
        crs=new_areas.crs,
    )
    assert expanded["area_id"].is_unique
    for split in ("val", "test"):
        before = original_areas.loc[
            original_areas.split == split, ["area_id", "geometry"]
        ].sort_values("area_id")
        after = expanded.loc[
            expanded.split == split, ["area_id", "geometry"]
        ].sort_values("area_id")
        if not before.reset_index(drop=True).equals(after.reset_index(drop=True)):
            raise AssertionError(f"{split} area geometries changed")
    destination_dir.mkdir(parents=True, exist_ok=True)
    new_path = destination_dir / "new_training_areas.parquet"
    all_path = destination_dir / "all_areas.parquet"
    # Avoid accidentally overwriting the saved selection on reruns.
    if new_path.exists() or all_path.exists():
        raise FileExistsError(
            f"Expanded area selection already exists in {destination_dir}. "
            "Inspect it before overwriting or choose another folder."
        )
    new_areas.to_parquet(new_path)
    expanded.to_parquet(all_path)
    print(f"Saved new training areas: {new_path}")
    print(f"Saved all areas: {all_path}")
    return expanded
