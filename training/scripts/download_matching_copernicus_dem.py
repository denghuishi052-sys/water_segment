from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


ROOT = Path(r"D:\project\water_segment")
SAMPLES_ROOT = ROOT / "data" / "processed" / "google_earth_flood_samples_global"
SAMPLES_INDEX = SAMPLES_ROOT / "samples_index.json"
CACHE_ROOT = ROOT / "data" / "raw" / "copernicus_dem_glo30"
DEM_INDEX = SAMPLES_ROOT / "dem_index.json"
BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_code(latitude: float, longitude: float) -> str:
    lat_degree = math.floor(latitude)
    lon_degree = math.floor(longitude)
    lat_prefix = "N" if lat_degree >= 0 else "S"
    lon_prefix = "E" if lon_degree >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_"
        f"{lat_prefix}{abs(lat_degree):02d}_00_"
        f"{lon_prefix}{abs(lon_degree):03d}_00_DEM"
    )


def tile_codes_for_bounds(
    left: float, bottom: float, right: float, top: float
) -> list[str]:
    last_latitude = math.floor(math.nextafter(top, -math.inf))
    last_longitude = math.floor(math.nextafter(right, -math.inf))
    return [
        tile_code(latitude, longitude)
        for latitude in range(math.floor(bottom), last_latitude + 1)
        for longitude in range(math.floor(left), last_longitude + 1)
    ]


def download_tile(code: str) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = CACHE_ROOT / f"{code}.tif"
    if destination.exists():
        return destination

    url = f"{BASE_URL}/{code}/{code}.tif"
    print(f"Downloading {url}")
    urlretrieve(url, destination)
    return destination


def write_preview(data: np.ndarray, output_path: Path) -> None:
    valid = data[np.isfinite(data)]
    lower, upper = np.percentile(valid, [2, 98])
    if lower == upper:
        upper = lower + 1

    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    image = ax.imshow(data, cmap="terrain", vmin=lower, vmax=upper)
    ax.set_axis_off()
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Elevation (m)")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_color_relief(
    data: np.ndarray,
    transform: rasterio.Affine,
    output_path: Path,
) -> None:
    valid = data[np.isfinite(data)]
    lower, upper = np.percentile(valid, [2, 98])
    if lower == upper:
        upper = lower + 1
    normalized = np.clip((data - lower) / (upper - lower), 0, 1)
    rgb = (plt.get_cmap("terrain")(normalized)[..., :3] * 255).astype(np.uint8)

    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "photometric": "RGB",
    }
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(np.moveaxis(rgb, 2, 0))
        output.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )
        output.update_tags(
            purpose="Display-ready color relief; use the float32 DEM for analysis",
            stretch_min_m=f"{lower:.6f}",
            stretch_max_m=f"{upper:.6f}",
        )


def process_sample(sample: dict) -> dict:
    bounds = sample["actual_bounds_wgs84"]
    left = bounds["left"]
    bottom = bounds["bottom"]
    right = bounds["right"]
    top = bounds["top"]

    codes = tile_codes_for_bounds(left, bottom, right, top)
    source_paths = [download_tile(code) for code in codes]
    width = max(1, math.ceil((right - left) * 3600))
    height = max(1, math.ceil((top - bottom) * 3600))
    transform = from_bounds(left, bottom, right, top, width, height)
    destination = np.full((height, width), np.nan, dtype=np.float32)

    sources = [rasterio.open(path) for path in source_paths]
    try:
        merged, merged_transform = merge(sources, bounds=(left, bottom, right, top))
        reproject(
            source=merged[0],
            destination=destination,
            src_transform=merged_transform,
            src_crs=sources[0].crs,
            src_nodata=sources[0].nodata,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    finally:
        for source in sources:
            source.close()

    sample_dir = SAMPLES_ROOT / sample["sample_id"]
    dem_path = sample_dir / f"{sample['sample_id']}_copernicus_dem_glo30.tif"
    color_relief_path = (
        sample_dir / f"{sample['sample_id']}_copernicus_dem_glo30_color_relief.tif"
    )
    preview_path = sample_dir / f"{sample['sample_id']}_copernicus_dem_glo30_preview.png"
    metadata_path = sample_dir / f"{sample['sample_id']}_copernicus_dem_glo30_metadata.json"

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(dem_path, "w", **profile) as destination_file:
        destination_file.write(destination, 1)
        destination_file.set_band_description(1, "Elevation above EGM2008 geoid (m)")
        valid = destination[np.isfinite(destination)]
        destination_file.update_tags(
            1,
            STATISTICS_MINIMUM=f"{valid.min():.9f}",
            STATISTICS_MAXIMUM=f"{valid.max():.9f}",
            STATISTICS_MEAN=f"{valid.mean():.9f}",
            STATISTICS_STDDEV=f"{valid.std():.9f}",
        )
        destination_file.update_tags(
            source="Copernicus DEM GLO-30",
            source_tiles=",".join(codes),
            vertical_reference="EGM2008 geoid",
            units="metre",
        )

    write_color_relief(destination, transform, color_relief_path)
    write_preview(destination, preview_path)
    metadata = {
        "sample_id": sample["sample_id"],
        "event": sample["event"],
        "imagery_date": sample["date"],
        "dem_source": "Copernicus DEM GLO-30",
        "source_tiles": codes,
        "horizontal_crs": "EPSG:4326",
        "vertical_reference": "EGM2008 geoid",
        "elevation_units": "metre",
        "bounds_wgs84": bounds,
        "width": width,
        "height": height,
        "resolution_degrees": [
            (right - left) / width,
            (top - bottom) / height,
        ],
        "elevation_min_m": float(valid.min()),
        "elevation_max_m": float(valid.max()),
        "elevation_mean_m": float(valid.mean()),
        "outputs": {
            "dem_geotiff": str(dem_path),
            "color_relief_geotiff": str(color_relief_path),
            "preview_png": str(preview_path),
            "metadata_json": str(metadata_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{sample['sample_id']}: {width}x{height}, "
        f"{metadata['elevation_min_m']:.2f} to {metadata['elevation_max_m']:.2f} m"
    )
    return metadata


def main() -> None:
    samples = json.loads(SAMPLES_INDEX.read_text(encoding="utf-8"))
    results = [process_sample(sample) for sample in samples]
    DEM_INDEX.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {DEM_INDEX}")


if __name__ == "__main__":
    main()
