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
