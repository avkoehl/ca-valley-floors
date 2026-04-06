import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pygeohydro.watershed import huc_wb_full


IGNORE_LIST = [
    "1805000506",  # farallon islands no flowlines
    "1807030501",
    "1807030502",
    "1807030503",
    "1807030504",
    "1807030506",
    "1807030507",
    "1710031206",  # oregon coast no flowlines
]

OCEAN_BAY_VALUES = {"Ocean", "Bay"}
SMALL_ISLAND_AREA_SQM = 10 * 1e6  # 10 sqkm in sqm

ARCGIS_URL = "https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/California_Cartographic_Coastal_Polygons/FeatureServer/31/query"
NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
)
US_STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_5m.zip"


def fetch_ocean_mask():
    print("Fetching California Cartographic Coastal Polygons...")
    features = []
    offset = 0
    page_size = 1000

    while True:
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        response = requests.get(ARCGIS_URL, params=params)
        response.raise_for_status()
        data = response.json()

        page_features = data.get("features", [])
        features.extend(page_features)
        print(f"  Fetched {len(features)} features so far...")

        if not data.get("properties", {}).get("exceededTransferLimit", False):
            break
        offset += page_size

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    print(f"  Total features fetched: {len(gdf)}")

    gdf_projected = gdf.to_crs("EPSG:3310")
    offshore = gdf["OFFSHORE"].fillna("")

    is_ocean_or_bay = offshore.isin(OCEAN_BAY_VALUES)
    is_small_island = (offshore == "") & (gdf_projected.area < SMALL_ISLAND_AREA_SQM)
    mask = gdf[is_ocean_or_bay | is_small_island].copy()

    print(f"  Ocean/bay features: {is_ocean_or_bay.sum()}")
    print(f"  Small island features filtered: {is_small_island.sum()}")
    print(f"  Total ocean mask features: {len(mask)}")

    return mask.to_crs("EPSG:3310")


def fetch_ca_boundary():
    print("Fetching US state boundaries...")
    us = gpd.read_file(US_STATES_URL)
    ca = us[us["NAME"] == "California"].copy()
    ca = ca.to_crs("EPSG:3310")
    print("  CA boundary fetched.")
    return ca


def fetch_na_boundary():
    print("Fetching Natural Earth North America boundary...")
    world = gpd.read_file(NATURAL_EARTH_URL)
    na = world[world["CONTINENT"] == "North America"].copy()
    na = na.to_crs("EPSG:3310")
    print(f"  North America boundary fetched ({len(na)} countries/territories).")
    return na


def build_target_hucs():
    print("Fetching HUC10 boundaries (this may take a while)...")
    huc10_boundaries = huc_wb_full(10)

    print("  Filtering to HUC Region 18 (California Region)...")
    watershed_huc10s = huc10_boundaries[huc10_boundaries["huc2"] == "18"]

    print("  Filtering to HUC10s intersecting California state boundary...")
    huc10_boundaries["states"] = huc10_boundaries["states"].fillna("")
    state_huc10s = huc10_boundaries[huc10_boundaries["states"].str.contains("CA")]

    combined = pd.concat([watershed_huc10s, state_huc10s])
    combined = combined.drop_duplicates(subset=["huc10"])
    combined = combined.rename(columns={"huc10": "hucid"})
    combined = combined[~combined["hucid"].isin(IGNORE_LIST)]

    print(f"  Total target HUC10s: {len(combined)}")
    return combined[["hucid", "states"]]


def main(data_dir: Path):
    prepare_dir = data_dir / "prepare"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    ocean_mask = fetch_ocean_mask()
    ocean_mask.to_file(prepare_dir / "ocean_mask.gpkg", driver="GPKG")
    print(f"  Saved ocean mask to {prepare_dir / 'ocean_mask.gpkg'}")

    ca_boundary = fetch_ca_boundary()
    ca_boundary.to_file(prepare_dir / "ca_boundary.gpkg", driver="GPKG")
    print(f"  Saved CA boundary to {prepare_dir / 'ca_boundary.gpkg'}")

    na_boundary = fetch_na_boundary()
    na_boundary.to_file(prepare_dir / "na_boundary.gpkg", driver="GPKG")
    print(f"  Saved NA boundary to {prepare_dir / 'na_boundary.gpkg'}")

    target_hucs = build_target_hucs()
    target_hucs.to_csv(prepare_dir / "target_hucs.csv", index=False)
    print(f"  Saved target HUCs to {prepare_dir / 'target_hucs.csv'}")

    print("\nPrepare complete.")
    print(f"  Ocean mask features:  {len(ocean_mask)}")
    print(f"  Target HUC10s:        {len(target_hucs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch boundary data and build target HUC10 list."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root data directory (default: ./data)",
    )
    args = parser.parse_args()
    main(args.data_dir)
