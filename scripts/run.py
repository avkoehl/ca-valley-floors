import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import importlib.metadata as meta
import pandas as pd
import rioxarray as rxr
from tqdm import tqdm

from valley_floor import delineate_from_dem_and_flowlines
from valley_floor.config import (
    Parameters,
    PreprocessingParameters,
    PostprocessingParameters,
)


# ---------------------------------------------------------------------------
# whitebox initialization
# ---------------------------------------------------------------------------


def _initialize_whitebox():
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    print(f"  WhiteboxTools binary: {wbt.exe_path}")


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------


def _get_package_info(package_name):
    try:
        dist = meta.distribution(package_name)
        version = dist.metadata["Version"]
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            commit = direct_url.get("vcs_info", {}).get("commit_id", "unknown")
            url = direct_url.get("url", "unknown")
            return {"url": url, "commit": commit, "version": version}
        return {"version": version}
    except Exception as e:
        return {"error": str(e)}


def write_manifest(output_dir, params, pre_params, post_params):
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "packages": {
            "valley-floor": _get_package_info("valley-floor"),
            "streamkit": _get_package_info("streamkit"),
        },
        "config": {
            "parameters": dataclasses.asdict(params),
            "preprocessing_parameters": dataclasses.asdict(pre_params),
            "postprocessing_parameters": dataclasses.asdict(post_params),
        },
    }
    out_path = output_dir / "run_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written to {out_path}")


# ---------------------------------------------------------------------------
# per-HUC worker (runs in subprocess)
# ---------------------------------------------------------------------------


def _process_huc(
    hucid, dem_path, flowlines_path, floors_dir, params, pre_params, post_params
):
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
    flowlines = gpd.read_file(flowlines_path)

    result = delineate_from_dem_and_flowlines(
        dem,
        flowlines,
        params=params,
        preprocessing_params=pre_params,
        postprocessing_params=post_params,
    )

    out_path = floors_dir / f"{hucid}_floor.tif"
    result["valley_floor"].rio.to_raster(
        out_path, dtype="uint8", compress="LZW", nodata=255
    )
    return hucid


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(data_dir: Path, max_workers: int, config_json: Path | None):
    prepare_dir = data_dir / "prepare"
    raw_dir = data_dir / "raw"
    floors_dir = data_dir / "floors"
    output_dir = data_dir / "output"
    floors_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load config
    if config_json is not None:
        with open(config_json) as f:
            cfg = json.load(f)
        params = Parameters(**cfg["parameters"])
        pre_params = PreprocessingParameters(**cfg["preprocessing_parameters"])
        post_params = PostprocessingParameters(**cfg["postprocessing_parameters"])
    else:
        params = Parameters()
        pre_params = PreprocessingParameters()
        post_params = PostprocessingParameters()

    # determine work
    target_hucs = pd.read_csv(prepare_dir / "target_hucs.csv", dtype=str)
    all_hucids = target_hucs["hucid"].tolist()

    to_process = []
    skipped = []
    missing_data = []

    for hucid in all_hucids:
        floor_path = floors_dir / f"{hucid}_floor.tif"
        if floor_path.exists():
            skipped.append(hucid)
            continue
        dem_path = raw_dir / f"{hucid}_dem.tif"
        fl_path = raw_dir / f"{hucid}_flowlines.gpkg"
        if not dem_path.exists() or not fl_path.exists():
            missing_data.append(hucid)
            continue
        to_process.append(hucid)

    print(f"\nRun summary:")
    print(f"  Total target HUCs:   {len(all_hucids)}")
    print(f"  Already done:        {len(skipped)}")
    print(f"  Missing raw data:    {len(missing_data)}")
    print(f"  To process:          {len(to_process)}")
    print(f"  Workers:             {max_workers}")
    print()

    if missing_data:
        print(f"  Warning: {len(missing_data)} HUCs skipped due to missing raw data.")
        print(
            f"  Run download.py first for: {missing_data[:5]}{'...' if len(missing_data) > 5 else ''}"
        )
        print()

    if not to_process:
        print("Nothing to process.")
    else:
        print("Initializing WhiteboxTools...")
        _initialize_whitebox()
        print()
        failed = []
        failed_path = floors_dir / "failed.txt"

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _process_huc,
                    hucid,
                    raw_dir / f"{hucid}_dem.tif",
                    raw_dir / f"{hucid}_flowlines.gpkg",
                    floors_dir,
                    params,
                    pre_params,
                    post_params,
                ): hucid
                for hucid in to_process
            }

            with tqdm(total=len(futures), desc="Processing HUCs", unit="huc") as pbar:
                for future in as_completed(futures):
                    hucid = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        failed.append((hucid, str(e)))
                        tqdm.write(f"  FAILED {hucid}: {e}")
                    pbar.update(1)
                    pbar.set_postfix(failed=len(failed))

        with open(failed_path, "w") as f:
            for hucid, reason in failed:
                f.write(f"{hucid}\t{reason}\n")

        print(f"\nProcessing complete.")
        print(f"  Succeeded: {len(to_process) - len(failed)}")
        print(f"  Failed:    {len(failed)}")
        if failed:
            print(f"  Failed HUCs written to {failed_path}")

    # write manifest if all expected floors exist
    all_floors_exist = all(
        (floors_dir / f"{hucid}_floor.tif").exists()
        for hucid in all_hucids
        if hucid not in missing_data
    )
    print()
    if all_floors_exist:
        print("All floors complete. Writing run manifest...")
        write_manifest(output_dir, params, pre_params, post_params)
        print("Ready to run mosaic.py")
    else:
        n_done = sum(
            1 for hucid in all_hucids if (floors_dir / f"{hucid}_floor.tif").exists()
        )
        print(
            f"Floors complete: {n_done}/{len(all_hucids) - len(missing_data)}. Re-run to continue."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delineate valley floors for all target HUC10s."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Path to a config JSON produced by a previous run (default: use package defaults)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.max_workers, args.config_json)
