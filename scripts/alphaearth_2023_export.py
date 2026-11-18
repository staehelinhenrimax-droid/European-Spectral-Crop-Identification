"""Export aligned 2023 AlphaEarth 64-band embeddings from Earth Engine.

Uses EXISTING 80-band Sentinel-2 area rasters as reference grids, so imagery
can later be concatenated to 144 channels without any spatial reprojection.

Earth Engine exports run on Google servers and write GeoTIFFs to Google Drive;
they are NOT automatically copied to your Mac.

Install in a Python environment: pip install earthengine-api rasterio numpy
Earth Engine requires Google authentication and an enabled Cloud project.

AlphaEarth attribution: The AlphaEarth Foundations Satellite Embedding dataset
is produced by Google and Google DeepMind.
"""

from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import rasterio

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
NAME_RE = re.compile(r"^germany_2023_area_(\d+)\.tif$")


def find_reference_images(reference_images_dir):
    root = Path(reference_images_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing Sentinel-2 reference image directory: {root}")
    files = sorted(p for p in root.glob("germany_2023_area_*.tif")
                   if NAME_RE.fullmatch(p.name))
    if not files:
        raise FileNotFoundError(f"No Germany area rasters found in {root}")
    ids = [int(NAME_RE.fullmatch(p.name).group(1)) for p in files]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate area IDs")
    return files


def reference_grid(path):
    path = Path(path)
    with rasterio.open(path) as src:
        if src.count != 80:
            raise ValueError(f"{path.name}: expected reference to contain 80 bands, got {src.count}")
        if src.crs is None:
            raise ValueError(f"{path}: CRS missing")
        t = src.transform
        if t.b != 0 or t.d != 0:
            raise ValueError(f"Rotated reference grid not supported: {path}")
        if t.a <= 0 or t.e >= 0:
            raise ValueError(f"Unexpected pixel orientation in {path}")
        crs = src.crs.to_string()
        grid = {
            "crs": crs,
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "width": src.width,
            "height": src.height,
            "bounds": [src.bounds.left, src.bounds.bottom,
                       src.bounds.right, src.bounds.top],
        }
    return grid


def authenticate_ee(cloud_project):
    import ee
    if not cloud_project or not str(cloud_project).strip():
        raise ValueError("Provide your Google Cloud project ID via cloud_project=...")
    try:
        ee.Initialize(project=cloud_project)
    except Exception:
        print("Earth Engine authentication required; follow the browser sign-in prompt")
        ee.Authenticate()
        ee.Initialize(project=cloud_project)
    return ee


def create_export_tasks(
    reference_images_dir,
    cloud_project,
    *,
    year=2023,
    drive_folder="SDS320_AlphaEarth_2023",
    max_tasks=1,
    dry_run=True,
    tasks_manifest=None,
):
    """Start up to max_tasks exports; default dry-run inspects grids only.

    To avoid accidentally submitting duplicates, existing tasks are NOT
    automatically retried or deduplicated. Inspect Tasks in EE before reruns.
    Use max_tasks=1 and dry_run=False for one test; then 50 after validation.
    """
    reference_paths = find_reference_images(reference_images_dir)
    if max_tasks < 1:
        raise ValueError("max_tasks must be >=1")
    if tasks_manifest is not None:
        mp = Path(tasks_manifest)
        if mp.exists():
            raise FileExistsError(f"Manifest already exists: {mp}; choose a new filename to prevent accidental repeat exports")
    selected = reference_paths[:max_tasks]
    plans = []
    for path in selected:
        plans.append({"area_id": int(NAME_RE.fullmatch(path.name).group(1)),
                      "output_filename": path.name,
                      "reference_path": str(path),
                      **reference_grid(path)})
    print(f"Found {len(reference_paths)} reference areas; selected {len(plans)}")
    print(f"Example area: {plans[0]['area_id']}, grid: {plans[0]['width']}x{plans[0]['height']}, CRS: {plans[0]['crs']}")
    print("Expected output: 64 float bands A00..A63, on exact Sentinel-2 grid")
    if dry_run:
        print("DRY RUN: No Earth Engine tasks submitted")
        return plans

    ee = authenticate_ee(cloud_project)
    collection = ee.ImageCollection(COLLECTION).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    # Compose tiles only within the year, preserving all 64 embedding dimensions.
    # EE masks missing coverage; exporting with a distinctive nodata value makes
    # it easy to detect incomplete areas in the validation step.
    for plan in plans:
        region = ee.Geometry.Rectangle(
            plan["bounds"], proj=plan["crs"], geodesic=False
        )
        image = (collection.filterBounds(region).mosaic()
                 .select(BANDS).toFloat().unmask(-9999))
        description = f"alphaearth_{year}_area_{plan['area_id']}"
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=drive_folder,
            fileNamePrefix=plan["output_filename"].removesuffix(".tif"),
            region=region,
            crs=plan["crs"],
            crsTransform=plan["transform"],
            maxPixels=1e9,
            fileFormat="GeoTIFF",
        )
        task.start()
        plan["task_id"] = task.id
        plan["task_state_at_submission"] = task.status().get("state")
        print(f"Started {description}: task {task.id}")
    if tasks_manifest is not None:
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"year": year, "drive_folder": drive_folder,
                                  "tasks": plans}, indent=2))
        print("Manifest saved:", mp)
    print("Export tasks submitted. Check Earth Engine Tasks / Drive for completion.")
    return plans


def validate_downloads(downloads_dir, reference_images_dir, *,
                       output_dir=None, allow_missing_pixels=False):
    """Validate exported GeoTIFFs and optionally copy to model-ready directory.

    Assumes each Drive-exported TIFF was downloaded unchanged and renamed to
    germany_2023_area_<id>.tif if necessary. No implicit resampling occurs.
    """
    import shutil

    downloads_dir = Path(downloads_dir)
    if not downloads_dir.is_dir():
        raise FileNotFoundError(downloads_dir)
    references = find_reference_images(reference_images_dir)
    destination = Path(output_dir) if output_dir else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for ref in references:
        exported = downloads_dir / ref.name
        if not exported.exists():
            reports.append({"area": ref.name, "status": "missing_download"})
            continue
        with rasterio.open(ref) as a, rasterio.open(exported) as b:
            errors = []
            if b.count != 64: errors.append(f"bands={b.count}, expected 64")
            if a.crs != b.crs: errors.append("CRS mismatch")
            if a.width != b.width or a.height != b.height:
                errors.append("shape mismatch")
            if not a.transform.almost_equals(b.transform):
                errors.append("transform mismatch")
            if errors:
                reports.append({"area": ref.name, "status": "; ".join(errors)})
                continue
            # Examine all pixels for nodata/incomplete coverage; 64 x 512 x 512
            # float32 (~67 MB) is reasonable per area.
            data = b.read()
            absent = np.any((data == -9999) | ~np.isfinite(data), axis=0)
            missing_fraction = float(absent.mean())
            if missing_fraction and not allow_missing_pixels:
                reports.append({"area": ref.name,
                                "status": f"missing embedding pixels ({missing_fraction:.3%})"})
                continue
        reports.append({"area": ref.name, "status": "ok",
                        "missing_fraction": missing_fraction})
        if destination:
            if exported.resolve() != (destination / ref.name).resolve():
                shutil.copy2(exported, destination / ref.name)
    ok = sum(row["status"] == "ok" for row in reports)
    print(f"Validated {ok}/{len(references)} reference areas successfully")
    for row in reports:
        if row["status"] != "ok": print(row)
    return reports
