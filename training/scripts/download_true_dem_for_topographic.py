from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


INPUT = Path(
    r"C:\Users\17473\Downloads\Browser_images (1)"
    r"\2026-07-23-00_00_2026-07-23-23_59_DEM_COPERNICUS_30_Topographic.tiff"
)
OUTPUT_DIR = INPUT.parent
CACHE_DIR = Path(r"D:\project\water_segment\data\raw\copernicus_dem_glo30")
BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_code(latitude: int, longitude: int) -> str:
    lat_prefix = "N" if latitude >= 0 else "S"
    lon_prefix = "E" if longitude >= 0 else "W"
    return (
        "Copernicus_DSM_COG_10_"
        f"{lat_prefix}{abs(latitude):02d}_00_"
        f"{lon_prefix}{abs(longitude):03d}_00_DEM"
    )


def tile_codes_for_bounds(
    left: float, bottom: float, right: float, top: float
) -> list[str]:
    max_lat = math.floor(math.nextafter(top, -math.inf))
    max_lon = math.floor(math.nextafter(right, -math.inf))
    return [
        tile_code(latitude, longitude)
        for latitude in range(math.floor(bottom), max_lat + 1)
        for longitude in range(math.floor(left), max_lon + 1)
    ]


def download_tile(code: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{code}.tif"
    if not path.exists():
        url = f"{BASE_URL}/{code}/{code}.tif"
        print(f"Downloading {url}")
        urlretrieve(url, path)
    return path


def write_dem(
    path: Path,
    data: np.ndarray,
    transform: rasterio.Affine,
    source_tiles: list[str],
    purpose: str,
) -> None:
    valid = data[np.isfinite(data)]
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as output:
        output.write(data.astype(np.float32), 1)
        output.set_band_description(1, "Elevation above EGM2008 geoid (m)")
        output.update_tags(
            source="Copernicus DEM GLO-30",
            source_tiles=",".join(source_tiles),
            vertical_reference="EGM2008 geoid",
            units="metre",
            purpose=purpose,
        )
        output.update_tags(
            1,
            STATISTICS_MINIMUM=f"{valid.min():.9f}",
            STATISTICS_MAXIMUM=f"{valid.max():.9f}",
            STATISTICS_MEAN=f"{valid.mean():.9f}",
            STATISTICS_STDDEV=f"{valid.std():.9f}",
        )


def main() -> None:
    with rasterio.open(INPUT) as display:
        bounds = display.bounds
        display_width = display.width
        display_height = display.height
        display_transform = display.transform

    codes = tile_codes_for_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    paths = [download_tile(code) for code in codes]
    sources = [rasterio.open(path) for path in paths]
    try:
        mosaic, mosaic_transform = merge(
            sources,
            bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        )
        source_crs = sources[0].crs
        source_nodata = sources[0].nodata
    finally:
        for source in sources:
            source.close()

    native_width = max(1, math.ceil((bounds.right - bounds.left) * 3600))
    native_height = max(1, math.ceil((bounds.top - bounds.bottom) * 3600))
    native_transform = from_bounds(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        native_width,
        native_height,
    )
    native = np.full((native_height, native_width), np.nan, dtype=np.float32)
    reproject(
        source=mosaic[0],
        destination=native,
        src_transform=mosaic_transform,
        src_crs=source_crs,
        src_nodata=source_nodata,
        dst_transform=native_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    native = fillnodata(
        native,
        mask=np.isfinite(native).astype(np.uint8),
        max_search_distance=10,
    ).astype(np.float32)

    aligned = np.full(
        (display_height, display_width), np.nan, dtype=np.float32
    )
    reproject(
        source=native,
        destination=aligned,
        src_transform=native_transform,
        src_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_transform=display_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    aligned = fillnodata(
        aligned,
        mask=np.isfinite(aligned).astype(np.uint8),
        max_search_distance=50,
    ).astype(np.float32)

    native_path = OUTPUT_DIR / "Copernicus_GLO30_true_DEM_native_30m.tif"
    aligned_path = OUTPUT_DIR / "Copernicus_GLO30_true_DEM_aligned_to_Topographic.tif"
    metadata_path = OUTPUT_DIR / "Copernicus_GLO30_true_DEM_metadata.json"
    write_dem(
        native_path,
        native,
        native_transform,
        codes,
        "Native-resolution DEM cropped to the Topographic TIFF bounds",
    )
    write_dem(
        aligned_path,
        aligned,
        display_transform,
        codes,
        "DEM resampled to exactly match the Topographic TIFF grid",
    )

    valid = native[np.isfinite(native)]
    metadata = {
        "source": "Copernicus DEM GLO-30",
        "source_tiles": codes,
        "input_topographic_tiff": str(INPUT),
        "crs": "EPSG:4326",
        "bounds": {
            "left": bounds.left,
            "bottom": bounds.bottom,
            "right": bounds.right,
            "top": bounds.top,
        },
        "native_size": [native_width, native_height],
        "aligned_size": [display_width, display_height],
        "elevation_units": "metre",
        "vertical_reference": "EGM2008 geoid",
        "elevation_min_m": float(valid.min()),
        "elevation_max_m": float(valid.max()),
        "elevation_mean_m": float(valid.mean()),
        "outputs": {
            "native_dem": str(native_path),
            "aligned_dem": str(aligned_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
