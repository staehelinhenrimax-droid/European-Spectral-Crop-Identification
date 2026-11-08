"""Load EuroCrops polygons and attach harmonized crop labels.

Use country-specific 'nuts' identifiers from EuroCrops.csv (e.g. 'dea').
The original polygon CRS and geometry are retained for individual datasets.
"""

from pathlib import Path
from collections.abc import Mapping

import geopandas as gpd
import pandas as pd


def load_crop_fields(
    vector_path: str | Path,
    lookup_path: str | Path,
    country_code: str,
    *,
    drop_unmatched: bool = False,
) -> gpd.GeoDataFrame:
    """Read a EuroCrops vector dataset and join its country-specific crop names.

    Parameters
    ----------
    vector_path : str or Path
        Polygon file (.parquet, .gpkg, .geojson, .shp, etc.).
    lookup_path : str or Path
        EuroCrops.csv containing 'nuts', 'original_code' and harmonized labels.
    country_code : str
        Exact 'nuts' country identifier from the lookup (e.g. 'dea').
    drop_unmatched : bool
        If True, discard fields without a harmonized label. Otherwise retain
        the polygons with missing labels, matching the original notebook.
    """
    vector_path = Path(vector_path)
    lookup_path = Path(lookup_path)
    if not vector_path.is_file():
        raise FileNotFoundError(vector_path)
    if not lookup_path.is_file():
        raise FileNotFoundError(lookup_path)

    if vector_path.suffix.lower() in {'.parquet', '.geoparquet'}:
        fields = gpd.read_parquet(vector_path)
    else:
        fields = gpd.read_file(vector_path)

    lookup = pd.read_csv(lookup_path)
    for col in ('nuts', 'original_code', 'hcat4_name'):
        if col not in lookup.columns:
            raise ValueError(f"Lookup file lacks required column {col!r}")
    if 'original_code' not in fields.columns:
        raise ValueError("Vector dataset lacks required column 'original_code'")

    country_lookup = lookup.loc[lookup['nuts'] == country_code].copy()
    if country_lookup.empty:
        raise ValueError(f"No EuroCrops lookup rows for nuts={country_code!r}")
    if country_lookup['original_code'].duplicated().any():
        raise ValueError(
            f"Duplicate original_code values in lookup for {country_code!r}; "
            "cannot safely join one label per polygon"
        )

    # Don't overwrite any preexisting columns from an earlier join.
    overlapping = (set(fields.columns) & set(country_lookup.columns)) - {'original_code'}
    if overlapping:
        raise ValueError(f"Fields already contain lookup columns: {sorted(overlapping)}")

    result = fields.merge(
        country_lookup,
        on='original_code',
        how='left',
        validate='many_to_one',
    )
    result = gpd.GeoDataFrame(result, geometry=fields.geometry.name, crs=fields.crs)
    if drop_unmatched:
        result = result.loc[result['hcat4_name'].notna()].copy()
    return result


def load_crop_fields_multi(
    datasets: Mapping[str, str | Path],
    lookup_path: str | Path,
    *,
    target_crs: str | None = None,
    drop_unmatched: bool = False,
) -> gpd.GeoDataFrame:
    """Combine multiple country datasets keyed by their EuroCrops 'nuts' codes.

    Specify target_crs when the input datasets have different CRSs. A
    'source_country' column records each polygon's lookup country code.
    """
    if not datasets:
        raise ValueError('At least one dataset is required')
    merged = []
    for code, path in datasets.items():
        fields = load_crop_fields(path, lookup_path, code, drop_unmatched=drop_unmatched)
        if target_crs is not None:
            fields = fields.to_crs(target_crs)
        fields = fields.assign(source_country=code)
        merged.append(fields)

    crs_list = [gdf.crs for gdf in merged]
    if len(merged) > 1 and len(set(map(str, crs_list))) > 1:
        raise ValueError('Datasets have different CRSs; supply target_crs to combine them')
    result = pd.concat(merged, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry=merged[0].geometry.name, crs=merged[0].crs)
