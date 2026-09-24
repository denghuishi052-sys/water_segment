"""Download an AOI-aligned Sentinel-2 L2A stack and Copernicus DEM from Earth Search.

The sources are public HTTPS Cloud Optimized GeoTIFFs.  No CDSE, AWS or OAuth
credentials are required.  The output grid is the native 10 m red-band grid;
all other assets, including the 30 m DEM, are warped onto that exact grid.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import rasterio
from pystac_client import Client
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds


STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-c1-l2a"
DEM_COLLECTION = "cop-dem-glo-30"
S2_ASSETS = {
    "B02_blue": ("blue", Resampling.bilinear),
    "B03_green": ("green", Resampling.bilinear),
    "B04_red": ("red", Resampling.bilinear),
    "B08_nir": ("nir", Resampling.bilinear),
    "B11_swir1": ("swir16", Resampling.bilinear),
    "B12_swir2": ("swir22", Resampling.bilinear),
    "SCL": ("scl", Resampling.nearest),
}
EARTH_SEARCH_S3_HOST = "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"
AWS_S3_PATH_HOST = "s3.us-west-2.amazonaws.com"
COP_DEM_BUCKET = "copernicus-dem-30m"
COP_DEM_S3_PATH_HOST = "s3.eu-central-1.amazonaws.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--start", required=True, help="Inclusive date, e.g. 2021-07-16")
    parser.add_argument("--end", required=True, help="Inclusive date, e.g. 2021-07-31")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--dem-local", type=Path, help="Optional local DEM GeoTIFF fallback when remote COG byte-range reads are blocked.")
    return parser.parse_args()


def cog_url(href: str) -> str:
    """Use the path-style public S3 endpoint when a network blocks virtual hosts."""
    parts = urlsplit(href)
    if parts.scheme == "s3" and parts.netloc == COP_DEM_BUCKET:
        return urlunsplit(("https", COP_DEM_S3_PATH_HOST, f"/{COP_DEM_BUCKET}{parts.path}", "", ""))
    if parts.hostname != EARTH_SEARCH_S3_HOST:
        return href
    return urlunsplit((parts.scheme, AWS_S3_PATH_HOST, f"/e84-earth-search-sentinel-data{parts.path}", parts.query, parts.fragment))


def rounded_window(dataset: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float]) -> Window:
    """Return the AOI window in ``dataset`` pixels, rejecting partial scenes."""
    native_bounds = transform_bounds("EPSG:4326", dataset.crs, *bounds, densify_pts=21)
    window = from_bounds(*native_bounds, transform=dataset.transform).round_offsets().round_lengths()
    full = Window(0, 0, dataset.width, dataset.height)
    if window.intersection(full) != window:
        raise RuntimeError("The selected Sentinel-2 tile does not cover the full AOI; use an AOI within one MGRS tile.")
    return window


def covers_bbox(item, bbox: list[float]) -> bool:
    """Whether an item's declared WGS84 extent fully contains the requested AOI."""
    if item.bbox is None:
        return False
    west, south, east, north = item.bbox
    requested_west, requested_south, requested_east, requested_north = bbox
    return west <= requested_west and south <= requested_south and east >= requested_east and north >= requested_north


def write_warped_asset(source: rasterio.io.DatasetReader, destination: Path, profile: dict, resampling: Resampling) -> None:
    data = np.zeros((profile["height"], profile["width"]), dtype=source.dtypes[0])
    reproject(
        source=rasterio.band(source, 1),
        destination=data,
        src_nodata=source.nodata,
        dst_nodata=0,
        dst_transform=profile["transform"],
        dst_crs=profile["crs"],
        resampling=resampling,
    )
    output_profile = profile | {"count": 1, "dtype": data.dtype, "nodata": 0, "compress": "deflate", "tiled": True}
    with rasterio.open(destination, "w", **output_profile) as target:
        target.write(data, 1)


def load_dem_aoi(source: rasterio.io.DatasetReader, target_profile: dict) -> tuple[np.ndarray, rasterio.Affine]:
    """Read only the source area capable of contributing to the target grid."""
    target_bounds = array_bounds(target_profile["height"], target_profile["width"], target_profile["transform"])
    source_bounds = transform_bounds(target_profile["crs"], source.crs, *target_bounds, densify_pts=21)
    window = from_bounds(*source_bounds, transform=source.transform).round_offsets().round_lengths()
    window = window.intersection(Window(0, 0, source.width, source.height))
    return source.read(1, window=window), source.window_transform(window)


def write_dem(sources: Iterable[tuple[str, str]], target_profile: dict, destination: Path) -> list[str]:
    nodata = np.float32(-9999.0)
    dem = np.full((target_profile["height"], target_profile["width"]), nodata, dtype=np.float32)
    item_ids: list[str] = []
    for source_id, source_href in sources:
        with rasterio.open(source_href) as source:
            data, transform = load_dem_aoi(source, target_profile)
            reproject(
                source=data,
                destination=dem,
                src_transform=transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
                dst_nodata=nodata,
                init_dest_nodata=False,
                resampling=Resampling.bilinear,
            )
        item_ids.append(source_id)
    if np.all(dem == nodata):
        raise RuntimeError("No DEM data intersected the target grid.")
    profile = target_profile | {"count": 1, "dtype": "float32", "nodata": nodata, "compress": "deflate", "tiled": True}
    with rasterio.open(destination, "w", **profile) as target:
        target.write(dem, 1)
    return item_ids


def write_rgb_preview(s2_directory: Path, destination: Path) -> None:
    """Create a contrast-stretched true-colour PNG for a fast visual check."""
    channels = []
    for name in ("B04_red.tif", "B03_green.tif", "B02_blue.tif"):
        with rasterio.open(s2_directory / name) as source:
            data = source.read(1).astype(np.float32)
        valid = data[data > 0]
        if valid.size:
            low, high = np.percentile(valid, (2, 98))
            if high > low:
                data = np.clip((data - low) * (255.0 / (high - low)), 0, 255)
        channels.append(data.astype(np.uint8))
    Image.fromarray(np.dstack(channels), mode="RGB").save(destination)


def main() -> None:
    args = parse_args()
    west, south, east, north = args.bbox
    if not west < east or not south < north:
        raise ValueError("--bbox must be WEST SOUTH EAST NORTH with west < east and south < north.")
    if not 0 <= args.max_cloud <= 100:
        raise ValueError("--max-cloud must be between 0 and 100.")

    # Improve remote COG reads, especially on public S3 endpoints.
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    catalog = Client.open(STAC_URL)
    bbox = [west, south, east, north]
    s2_items = list(catalog.search(
        collections=[S2_COLLECTION], bbox=bbox, datetime=f"{args.start}/{args.end}",
        query={"eo:cloud_cover": {"lte": args.max_cloud}}, max_items=args.max_items,
    ).items())
    if not s2_items:
        raise RuntimeError("No Sentinel-2 scenes matched the AOI, date range and cloud limit.")
    complete_s2_items = [item for item in s2_items if covers_bbox(item, bbox)]
    if not complete_s2_items:
        raise RuntimeError("No single Sentinel-2 tile completely covers this AOI; split the AOI or implement a multi-tile mosaic.")
    best_s2 = min(complete_s2_items, key=lambda item: item.properties.get("eo:cloud_cover", 100.0))
    missing = [asset for asset, (key, _) in S2_ASSETS.items() if key not in best_s2.assets]
    if missing:
        raise RuntimeError(f"Selected Sentinel-2 item lacks assets: {', '.join(missing)}")

    output = args.output_dir
    s2_output, dem_output = output / "sentinel2", output / "dem"
    s2_output.mkdir(parents=True, exist_ok=True)
    dem_output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        red = stack.enter_context(rasterio.open(cog_url(best_s2.assets["red"].href)))
        window = rounded_window(red, tuple(bbox))
        profile = red.profile.copy()
        profile.update(height=int(window.height), width=int(window.width), transform=red.window_transform(window), driver="GTiff")
        for filename, (asset_key, resampling) in S2_ASSETS.items():
            source = stack.enter_context(rasterio.open(cog_url(best_s2.assets[asset_key].href)))
            write_warped_asset(source, s2_output / f"{filename}.tif", profile, resampling)

    dem_items = list(catalog.search(collections=[DEM_COLLECTION], bbox=bbox, max_items=args.max_items).items())
    if args.dem_local:
        if not args.dem_local.is_file():
            raise FileNotFoundError(f"Local DEM does not exist: {args.dem_local}")
        dem_sources = [(args.dem_local.name, str(args.dem_local))]
    else:
        if not dem_items:
            raise RuntimeError("No Copernicus DEM tiles intersect the AOI.")
        dem_sources = [(item.id, cog_url(item.assets["data"].href)) for item in dem_items if "data" in item.assets]
    dem_ids = write_dem(dem_sources, profile, dem_output / "copernicus_dem.tif")
    write_rgb_preview(s2_output, output / "sentinel2_rgb_preview.png")
    metadata = {
        "stac_url": STAC_URL, "bbox_wgs84": bbox, "sentinel2_item": best_s2.id,
        "sentinel2_datetime": str(best_s2.properties.get("datetime")),
        "sentinel2_cloud_cover": best_s2.properties.get("eo:cloud_cover"),
        "dem_items": dem_ids, "grid_crs": str(profile["crs"]),
        "grid_width": profile["width"], "grid_height": profile["height"],
        "grid_transform": list(profile["transform"]),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
