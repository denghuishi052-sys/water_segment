from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds


ROOT = Path(r"D:\project\water_segment")
SAMPLES_ROOT = ROOT / "data" / "processed" / "google_earth_flood_samples_global"
SAMPLES_INDEX = SAMPLES_ROOT / "samples_index.json"
OUTPUT_INDEX = SAMPLES_ROOT / "usgs_3dep_1m_dem_index.json"

SOURCES = {
    "houston_harvey_addicks_2017": {
        "url": (
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/"
            "Projects/TX_CoastalRegion_2018_A18/TIFF/"
            "USGS_1M_15_x24y330_TX_CoastalRegion_2018_A18.tif"
        ),
        "project": "TX_CoastalRegion_2018_A18",
        "acquisition": "2018",
    },
    "midland_michigan_dam_flood_2020": {
        "url": (
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/"
            "Projects/MI_31Co_Midland_2016/TIFF/"
            "USGS_one_meter_x72y484_MI_31Co_Midland_2016.tif"
        ),
        "project": "MI_31Co_Midland_2016",
        "acquisition": "2016",
    },
}


def clipped_window(dataset: rasterio.DatasetReader, bounds: tuple[float, ...]) -> Window:
    window = from_bounds(*bounds, transform=dataset.transform)
    window = window.round_offsets().round_lengths()
    return window.intersection(Window(0, 0, dataset.width, dataset.height))


def write_dem(
    output_path: Path,
    data: np.ndarray,
    crs: rasterio.CRS | str,
    transform: rasterio.Affine,
    source_url: str,
) -> None:
    valid = data[np.isfinite(data)]
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(data.astype(np.float32), 1)
        output.set_band_description(1, "Bare-earth elevation (m)")
        output.update_tags(
            source="USGS 3DEP 1-meter DEM",
            source_url=source_url,
            vertical_reference="NAVD88",
            units="metre",
        )
        output.update_tags(
            1,
            STATISTICS_MINIMUM=f"{valid.min():.9f}",
            STATISTICS_MAXIMUM=f"{valid.max():.9f}",
            STATISTICS_MEAN=f"{valid.mean():.9f}",
            STATISTICS_STDDEV=f"{valid.std():.9f}",
        )


def hillshade(data: np.ndarray, pixel_size: float = 1.0) -> np.ndarray:
    filled = np.where(np.isfinite(data), data, np.nanmedian(data))
    dy, dx = np.gradient(filled, pixel_size, pixel_size)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth = np.deg2rad(315)
    altitude = np.deg2rad(45)
    shaded = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return (255 * (shaded - shaded.min()) / np.ptp(shaded)).astype(np.uint8)


def write_hillshade(
    output_path: Path,
    data: np.ndarray,
    crs: rasterio.CRS,
    transform: rasterio.Affine,
) -> None:
    shaded = hillshade(data)
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(shaded, 1)
        output.set_band_description(1, "Hillshade")


def write_preview(
    output_path: Path,
    data: np.ndarray,
    bounds: rasterio.coords.BoundingBox,
) -> None:
    shaded = hillshade(data)
    valid = data[np.isfinite(data)]
    low, high = np.percentile(valid, [2, 98])
    normalized = np.clip((data - low) / max(high - low, 1e-6), 0, 1)
    colors = plt.get_cmap("terrain")(normalized)[..., :3]
    relief = np.clip(colors * (0.35 + 0.65 * shaded[..., None] / 255), 0, 1)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    ax.imshow(relief)
    ax.set_title(
        f"USGS 3DEP 1 m DEM | {valid.min():.2f}–{valid.max():.2f} m",
        fontsize=12,
    )
    ax.set_xlabel(
        f"Native bounds: {bounds.left:.1f}, {bounds.bottom:.1f}, "
        f"{bounds.right:.1f}, {bounds.top:.1f}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process(sample: dict, source_config: dict) -> dict:
    bounds = sample["actual_bounds_wgs84"]
    wgs84_bounds = (
        bounds["left"],
        bounds["bottom"],
        bounds["right"],
        bounds["top"],
    )
    sample_dir = SAMPLES_ROOT / sample["sample_id"]
    prefix = sample_dir / f"{sample['sample_id']}_usgs_3dep_1m"
    native_path = Path(f"{prefix}_native_utm.tif")
    wgs84_path = Path(f"{prefix}_wgs84.tif")
    hillshade_path = Path(f"{prefix}_hillshade.tif")
    preview_path = Path(f"{prefix}_preview.png")
    metadata_path = Path(f"{prefix}_metadata.json")

    with rasterio.open(source_config["url"]) as source:
        source_bounds = transform_bounds(
            "EPSG:4326", source.crs, *wgs84_bounds, densify_pts=21
        )
        window = clipped_window(source, source_bounds)
        native = source.read(1, window=window).astype(np.float32)
        native[native == source.nodata] = np.nan
        native_transform = source.window_transform(window)
        native_crs = source.crs
        native_bounds = rasterio.transform.array_bounds(
            native.shape[0], native.shape[1], native_transform
        )

    write_dem(
        native_path, native, native_crs, native_transform, source_config["url"]
    )
    write_hillshade(hillshade_path, native, native_crs, native_transform)
    write_preview(
        preview_path,
        native,
        rasterio.coords.BoundingBox(*native_bounds),
    )

    mid_latitude = (bounds["bottom"] + bounds["top"]) / 2
    width = max(
        1,
        round(
            (bounds["right"] - bounds["left"])
            * 111_320
            * math.cos(math.radians(mid_latitude))
        ),
    )
    height = max(1, round((bounds["top"] - bounds["bottom"]) * 111_320))
    wgs84_transform = rasterio.transform.from_bounds(
        *wgs84_bounds, width, height
    )
    wgs84 = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=native,
        destination=wgs84,
        src_transform=native_transform,
        src_crs=native_crs,
        src_nodata=np.nan,
        dst_transform=wgs84_transform,
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    write_dem(
        wgs84_path, wgs84, "EPSG:4326", wgs84_transform, source_config["url"]
    )

    valid = native[np.isfinite(native)]
    metadata = {
        "sample_id": sample["sample_id"],
        "event": sample["event"],
        "imagery_date": sample["date"],
        "dem_source": "USGS 3DEP 1-meter DEM",
        "source_project": source_config["project"],
        "source_acquisition": source_config["acquisition"],
        "source_url": source_config["url"],
        "vertical_reference": "NAVD88",
        "elevation_units": "metre",
        "imagery_bounds_wgs84": bounds,
        "native_crs": str(native_crs),
        "native_resolution_m": list(map(float, (1.0, 1.0))),
        "native_width": native.shape[1],
        "native_height": native.shape[0],
        "wgs84_width": width,
        "wgs84_height": height,
        "elevation_min_m": float(valid.min()),
        "elevation_max_m": float(valid.max()),
        "elevation_mean_m": float(valid.mean()),
        "outputs": {
            "native_utm_geotiff": str(native_path),
            "exact_bounds_wgs84_geotiff": str(wgs84_path),
            "hillshade_geotiff": str(hillshade_path),
            "preview_png": str(preview_path),
            "metadata_json": str(metadata_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{sample['sample_id']}: native {native.shape[1]}x{native.shape[0]} "
        f"at 1 m; {valid.min():.2f} to {valid.max():.2f} m"
    )
    return metadata


def main() -> None:
    samples = {
        sample["sample_id"]: sample
        for sample in json.loads(SAMPLES_INDEX.read_text(encoding="utf-8"))
    }
    results = [
        process(samples[sample_id], source_config)
        for sample_id, source_config in SOURCES.items()
    ]
    OUTPUT_INDEX.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
