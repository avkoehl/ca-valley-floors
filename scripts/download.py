import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
import pandas as pd
from tqdm import tqdm

from streamkit.datasets import (
    download_3dep_dem,
    download_nhd_flowlines,
    download_wbd_boundary,
)

BUFFER_M = 100
CRS = "EPSG:3310"


def download_dem(hucid, huc_boundary, ocean_mask, raw_dir):
    out_path = raw_dir / f"{hucid}_dem.tif"
    if out_path.exists():
        return

    # buffer in 3310, reproject back to 4326 for download
    buffered = huc_boundary.to_crs("EPSG:3310").buffer(BUFFER_M)
    buffered = buffered.to_crs("EPSG:4326")

    dem = download_3dep_dem(buffered, resolution=10, crs="EPSG:4326")

    # clip ocean
    ocean = ocean_mask.to_crs("EPSG:4326")
    dem = dem.rio.clip(ocean.geometry, invert=True, all_touched=True, drop=False)

    # reproject to 3310
    dem = dem.rio.reproject(CRS, resampling=rasterio.enums.Resampling.bilinear)

    dem.rio.to_raster(out_path, compress="LZW", dtype="float32")


def download_flowlines(hucid, huc_boundary, raw_dir):
    out_path = raw_dir / f"{hucid}_flowlines.gpkg"
    if out_path.exists():
        return

    flowlines = download_nhd_flowlines(
        huc_boundary, layer="medium", linestring_only=True, crs=CRS
    )

    if flowlines.empty:
        raise ValueError(f"No flowlines found for HUC {hucid}")

    flowlines.to_file(out_path, driver="GPKG")


def process_huc(hucid, ocean_mask, raw_dir):
    dem_path = raw_dir / f"{hucid}_dem.tif"
    fl_path = raw_dir / f"{hucid}_flowlines.gpkg"

    if dem_path.exists() and fl_path.exists():
        return "skipped"

    huc_boundary = download_wbd_boundary(hucid)

    if not dem_path.exists():
        download_dem(hucid, huc_boundary, ocean_mask, raw_dir)

    if not fl_path.exists():
        download_flowlines(hucid, huc_boundary, raw_dir)

    return "done"


def main(data_dir: Path):
    prepare_dir = data_dir / "prepare"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_hucs = pd.read_csv(prepare_dir / "target_hucs.csv", dtype=str)
    ocean_mask = gpd.read_file(prepare_dir / "ocean_mask.gpkg")

    failed_path = raw_dir / "failed.txt"
    # load previously failed hucs so we retry them
    previously_failed = set()
    if failed_path.exists():
        with open(failed_path) as f:
            previously_failed = {
                line.strip().split("\t")[0] for line in f if line.strip()
            }

    hucids = target_hucs["hucid"].tolist()

    skipped = 0
    done = 0
    failed = []

    for hucid in tqdm(hucids, desc="Downloading HUCs"):
        try:
            status = process_huc(hucid, ocean_mask, raw_dir)
            if status == "skipped":
                skipped += 1
            else:
                done += 1
        except Exception as e:
            failed.append((hucid, str(e)))
            tqdm.write(f"FAILED {hucid}: {e}")

    # rewrite failed.txt with current failures only (clears previously resolved)
    with open(failed_path, "w") as f:
        for hucid, reason in failed:
            f.write(f"{hucid}\t{reason}\n")

    print(f"\nDownload complete.")
    print(f"  Done:    {done}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        print(f"  Failed HUCs written to {failed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download DEMs and flowlines for all target HUC10s."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root data directory (default: ./data)",
    )
    args = parser.parse_args()
    main(args.data_dir)
