from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import requests
import rasterio
from PIL import Image


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
OPENEO_RESULT_URL = "https://openeosh.dataspace.copernicus.eu/1.2/result"
CRS84 = "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
EARTH_RADIUS_M = 6_371_008.8
SENTINEL2_EARLIEST_DATE = date(2015, 6, 27)

S2_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B03", "B02"],
      units: ["DN", "DN", "DN"]
    }],
    output: {
      id: "default",
      bands: 3,
      sampleType: SampleType.UINT16
    }
  };
}

function evaluatePixel(sample) {
  return [sample.B04, sample.B03, sample.B02];
}
""".strip()

DEM_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["DEM"],
    output: {
      id: "default",
      bands: 1,
      sampleType: SampleType.FLOAT32
    }
  };
}

function evaluatePixel(sample) {
  return [sample.DEM];
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download matching Sentinel-2 L2A imagery and Copernicus GLO-30 "
            "DEM GeoTIFFs from Copernicus Data Space."
        )
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
        help="WGS84 longitude/latitude bounds.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="End date in YYYY-MM-DD format; defaults to start date.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=10.0,
        help=(
            "Approximate imagery pixel size in metres; values below the "
            "Sentinel-2 native maximum of 10 m are rejected (default: 10)."
        ),
    )
    parser.add_argument(
        "--dem-native-resolution",
        type=float,
        default=30.0,
        help="Approximate native DEM output pixel size in metres (default: 30).",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=50.0,
        help="Maximum Sentinel-2 tile cloud coverage percent (default: 50).",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=2500,
        help="Reject requests larger than this width or height (default: 2500).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for GeoTIFF, preview, request, and metadata files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write request JSON files without authenticating or downloading.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    west, south, east, north = args.bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid bbox; expected WEST SOUTH EAST NORTH in WGS84.")
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date or args.start_date)
    if start < SENTINEL2_EARLIEST_DATE:
        raise ValueError(
            "Sentinel-2 imagery is unavailable before 2015-06-27."
        )
    if end < start:
        raise ValueError("end-date must not be earlier than start-date.")
    if args.resolution <= 0 or args.dem_native_resolution <= 0:
        raise ValueError("Resolutions must be positive.")
    if args.resolution < 10:
        raise ValueError(
            "Sentinel-2 B02/B03/B04/B08 native resolution is 10 m. "
            "A smaller value would only upsample the image without adding detail."
        )
    if not 0 <= args.max_cloud <= 100:
        raise ValueError("max-cloud must be between 0 and 100.")


def dimensions_for_resolution(
    bbox: list[float], resolution_m: float
) -> tuple[int, int]:
    west, south, east, north = bbox
    mid_latitude = math.radians((south + north) / 2)
    width_m = (
        EARTH_RADIUS_M
        * math.radians(east - west)
        * math.cos(mid_latitude)
    )
    height_m = EARTH_RADIUS_M * math.radians(north - south)
    return max(1, math.ceil(width_m / resolution_m)), max(
        1, math.ceil(height_m / resolution_m)
    )


def common_output(width: int, height: int) -> dict[str, Any]:
    return {
        "width": width,
        "height": height,
        "responses": [
            {
                "identifier": "default",
                "format": {"type": "image/tiff"},
            }
        ],
    }


def imagery_request(
    bbox: list[float],
    start_date: str,
    end_date: str,
    width: int,
    height: int,
    max_cloud: float,
) -> dict[str, Any]:
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": CRS84},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date}T00:00:00Z",
                            "to": f"{end_date}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": max_cloud,
                    },
                    "processing": {
                        "upsampling": "BICUBIC",
                        "downsampling": "BICUBIC",
                    },
                }
            ],
        },
        "output": common_output(width, height),
        "evalscript": S2_EVALSCRIPT,
    }


def dem_request(
    bbox: list[float], width: int, height: int
) -> dict[str, Any]:
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": CRS84},
            },
            "data": [
                {
                    "type": "dem",
                    "dataFilter": {"demInstance": "COPERNICUS_30"},
                    "processing": {
                        "upsampling": "BILINEAR",
                        "downsampling": "BILINEAR",
                    },
                }
            ],
        },
        "output": common_output(width, height),
        "evalscript": DEM_EVALSCRIPT,
    }


def openeo_true_color_request(
    bbox: list[float],
    start_date: str,
    end_date: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    west, south, east, north = bbox
    return {
        "process": {
            "process_graph": {
                "load2": {
                    "process_id": "load_collection",
                    "arguments": {
                        "id": "sentinel-2-l2a",
                        "spatial_extent": {
                            "west": west,
                            "south": south,
                            "east": east,
                            "north": north,
                            "width": width,
                            "height": height,
                        },
                        "temporal_extent": [
                            f"{start_date}T00:00:00Z",
                            f"{end_date}T23:59:59Z",
                        ],
                        "bands": ["B04", "B03", "B02"],
                        "upsampling": "BICUBIC",
                        "downsampling": "BICUBIC",
                    },
                },
                "trueColor": {
                    "process_id": "true_color",
                    "arguments": {
                        "data": {"from_node": "load2"},
                        "maxR": 3,
                        "midR": 0.13,
                        "sat": 1.2,
                        "gamma": 1.8,
                        "gOff": 0.01,
                        "red": "B04",
                        "green": "B03",
                        "blue": "B02",
                    },
                },
                "save5": {
                    "process_id": "save_result",
                    "arguments": {
                        "format": "webp",
                        "data": {"from_node": "trueColor"},
                    },
                    "result": True,
                },
            },
            "parameters": [],
        }
    }


def get_access_token(client_id: str, client_secret: str) -> str:
    response = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=60,
            )
            if response.ok:
                break
        except requests.RequestException:
            if attempt == 4:
                raise
        if attempt < 4:
            time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError("OAuth token endpoint returned no response.")
    if not response.ok:
        raise RuntimeError(
            f"OAuth token request failed ({response.status_code}): "
            f"{response.text[:500]}"
        )
    return response.json()["access_token"]


def completed_raster(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as dataset:
            return dataset.width > 0 and dataset.height > 0
    except rasterio.errors.RasterioIOError:
        return False


def process_download(
    request_body: dict[str, Any],
    output_path: Path,
    token: str,
) -> None:
    if completed_raster(output_path):
        print(f"Reusing completed file: {output_path.name}")
        return

    response = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                PROCESS_URL,
                headers={"Authorization": f"Bearer {token}"},
                json=request_body,
                timeout=300,
                stream=True,
            )
            if response.ok:
                break
            if response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(
                    f"Process API request failed ({response.status_code}): "
                    f"{response.text[:1000]}"
                )
        except requests.RequestException:
            if attempt == 4:
                raise
        if attempt == 4:
            raise RuntimeError(
                f"Process API request failed ({response.status_code}): "
                f"{response.text[:1000]}"
            )
        delay = 2 ** attempt
        print(f"Transient API error; retrying in {delay} seconds...")
        time.sleep(delay)

    if response is None:
        raise RuntimeError("Process API returned no response.")
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    with temporary_path.open("wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    temporary_path.replace(output_path)


def openeo_download(
    request_body: dict[str, Any],
    output_path: Path,
    token: str,
) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Reusing completed file: {output_path.name}")
        return
    response = None
    for attempt in range(1, 5):
        try:
            response = requests.post(
                OPENEO_RESULT_URL,
                headers={"Authorization": f"Bearer {token}"},
                json=request_body,
                timeout=300,
            )
            if response.ok:
                break
            if response.status_code < 500 and response.status_code != 429:
                raise RuntimeError(
                    f"openEO request failed ({response.status_code}): "
                    f"{response.text[:1000]}"
                )
        except requests.RequestException:
            if attempt == 4:
                raise
        if attempt == 4:
            raise RuntimeError(
                f"openEO request failed ({response.status_code}): "
                f"{response.text[:1000]}"
            )
        time.sleep(2 ** attempt)
    if response is None:
        raise RuntimeError("openEO returned no response.")
    output_path.write_bytes(response.content)


def add_raster_metadata(
    imagery_path: Path,
    aligned_dem_path: Path,
    native_dem_path: Path,
) -> None:
    with rasterio.open(imagery_path, "r+") as imagery:
        expected_bands = ["Red (B04)", "Green (B03)", "Blue (B02)"]
        if imagery.count != len(expected_bands):
            raise RuntimeError(
                f"Expected 3 true-color imagery bands, received {imagery.count}."
            )
        for index, description in enumerate(expected_bands, start=1):
            imagery.set_band_description(index, description)
        imagery.update_tags(
            source="Copernicus Data Space Sentinel-2 L2A",
            band_order="B04,B03,B02",
            color_space="True color RGB",
            spectral_units="Sentinel-2 L2A digital numbers",
        )

    for path, purpose in [
        (aligned_dem_path, "Pixel-aligned to Sentinel-2 output"),
        (native_dem_path, "Approximate native 30 m output"),
    ]:
        with rasterio.open(path, "r+") as dem:
            if dem.count != 1:
                raise RuntimeError(
                    f"Expected one DEM band in {path.name}, received {dem.count}."
                )
            dem.set_band_description(1, "Elevation above EGM2008 geoid (m)")
            dem.update_tags(
                source="Copernicus DEM GLO-30 via Copernicus Data Space",
                units="metre",
                vertical_reference="EGM2008 geoid",
                purpose=purpose,
            )


def verify_matching_grid(
    imagery_path: Path, aligned_dem_path: Path
) -> dict[str, Any]:
    with rasterio.open(imagery_path) as imagery, rasterio.open(
        aligned_dem_path
    ) as dem:
        checks = {
            "same_width": imagery.width == dem.width,
            "same_height": imagery.height == dem.height,
            "same_crs": imagery.crs == dem.crs,
            "same_transform": imagery.transform.almost_equals(dem.transform),
            "same_bounds": all(
                math.isclose(a, b, abs_tol=1e-10)
                for a, b in zip(imagery.bounds, dem.bounds)
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Imagery and aligned DEM grids differ: {checks}")
        return {
            **checks,
            "width": imagery.width,
            "height": imagery.height,
            "crs": str(imagery.crs),
            "transform": list(imagery.transform),
            "bounds": list(imagery.bounds),
        }


def write_rgb_preview(
    imagery_path: Path,
    preview_path: Path,
    webp_path: Path,
) -> None:
    with rasterio.open(imagery_path) as source:
        rgb = source.read([1, 2, 3]).astype(np.float32)
    preview = true_color_uint8(rgb)
    image = Image.fromarray(np.moveaxis(preview, 0, 2), mode="RGB")
    image.save(preview_path)
    image.save(webp_path, format="WEBP", quality=95, method=6)


def true_color_uint8(rgb_dn: np.ndarray) -> np.ndarray:
    mask = np.any(rgb_dn > 0, axis=0)
    reflectance = rgb_dn.astype(np.float64) / 10_000.0
    max_reflectance = 3.0
    mid_reflectance = 0.13
    saturation = 1.2
    gamma = 1.8
    gamma_offset = 0.01

    scaled = np.clip(reflectance / max_reflectance, 0, 1)
    tx_scaled = mid_reflectance / max_reflectance
    numerator = scaled * (
        scaled * (tx_scaled + 1.0 - 1.0) - 1.0
    )
    denominator = scaled * (2.0 * tx_scaled - 1.0) - tx_scaled
    contrast = numerator / denominator

    offset_power = gamma_offset**gamma
    offset_range = (1.0 + gamma_offset) ** gamma - offset_power
    gamma_adjusted = (
        np.power(contrast + gamma_offset, gamma) - offset_power
    ) / offset_range

    channel_average = np.mean(gamma_adjusted, axis=0)
    average_shift = channel_average * (1.0 - saturation)
    saturated = np.clip(
        average_shift[None, :, :] + gamma_adjusted * saturation,
        0,
        1,
    )
    srgb = np.where(
        saturated <= 0.0031308,
        12.92 * saturated,
        1.055 * np.power(saturated, 1 / 2.4) - 0.055,
    )
    output = np.round(255 * np.clip(srgb, 0, 1)).astype(np.uint8)
    output[:, ~mask] = 0
    return output


def write_display_geotiff(
    imagery_path: Path,
    display_path: Path,
) -> None:
    with rasterio.open(imagery_path) as source:
        rgb = source.read([1, 2, 3])
        profile = source.profile.copy()
        profile.update(
            dtype="uint8",
            count=3,
            nodata=None,
            photometric="RGB",
            compress="deflate",
        )
        transform = source.transform
        crs = source.crs

    display = true_color_uint8(rgb)
    with rasterio.open(display_path, "w", **profile) as output:
        output.write(display)
        output.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )
        output.set_band_description(1, "Red display channel from B04")
        output.set_band_description(2, "Green display channel from B03")
        output.set_band_description(3, "Blue display channel from B02")
        output.update_tags(
            source=str(imagery_path),
            purpose="Copernicus optimized Sentinel-2 true color",
            scaling=(
                "openEO true_color: maxR=3, midR=0.13, sat=1.2, "
                "gamma=1.8, gOff=0.01, sRGB encoding"
            ),
            original_crs=str(crs),
            original_transform=",".join(map(str, transform)),
        )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f"Argument error: {error}", file=sys.stderr)
        return 2

    args.end_date = args.end_date or args.start_date
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imagery_width, imagery_height = dimensions_for_resolution(
        args.bbox, args.resolution
    )
    dem_width, dem_height = dimensions_for_resolution(
        args.bbox, args.dem_native_resolution
    )
    if max(imagery_width, imagery_height, dem_width, dem_height) > args.max_dimension:
        print(
            "Requested output exceeds max-dimension. Increase --resolution, "
            "reduce the bbox, or explicitly increase --max-dimension.",
            file=sys.stderr,
        )
        return 2

    date_label = (
        args.start_date
        if args.start_date == args.end_date
        else f"{args.start_date}_to_{args.end_date}"
    )
    imagery_payload = imagery_request(
        args.bbox,
        args.start_date,
        args.end_date,
        imagery_width,
        imagery_height,
        args.max_cloud,
    )
    aligned_dem_payload = dem_request(
        args.bbox, imagery_width, imagery_height
    )
    native_dem_payload = dem_request(args.bbox, dem_width, dem_height)
    openeo_payload = openeo_true_color_request(
        args.bbox,
        args.start_date,
        args.end_date,
        imagery_width,
        imagery_height,
    )

    imagery_request_path = args.output_dir / "request_sentinel2_l2a.json"
    aligned_dem_request_path = args.output_dir / "request_dem_aligned.json"
    native_dem_request_path = args.output_dir / "request_dem_native30m.json"
    openeo_request_path = args.output_dir / "request_openeo_true_color.json"
    write_json(imagery_request_path, imagery_payload)
    write_json(aligned_dem_request_path, aligned_dem_payload)
    write_json(native_dem_request_path, native_dem_payload)
    write_json(openeo_request_path, openeo_payload)

    plan = {
        "bbox_wgs84": args.bbox,
        "date_range": [args.start_date, args.end_date],
        "imagery_resolution_m_approx": args.resolution,
        "imagery_grid": [imagery_width, imagery_height],
        "native_dem_resolution_m_approx": args.dem_native_resolution,
        "native_dem_grid": [dem_width, dem_height],
        "max_cloud_percent": args.max_cloud,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"Dry run complete. Requests written to {args.output_dir}")
        return 0

    imagery_path = (
        args.output_dir
        / f"sentinel2_l2a_true_color_B4_B3_B2_{date_label}_"
        f"{args.resolution:g}m_bicubic.tif"
    )
    aligned_dem_path = (
        args.output_dir
        / f"copernicus_dem_glo30_aligned_{args.resolution:g}m.tif"
    )
    native_dem_path = (
        args.output_dir
        / f"copernicus_dem_glo30_native_{args.dem_native_resolution:g}m.tif"
    )
    preview_path = (
        args.output_dir
        / f"sentinel2_l2a_true_color_B4_B3_B2_{date_label}_preview.png"
    )
    display_path = (
        args.output_dir
        / f"sentinel2_l2a_true_color_B4_B3_B2_{date_label}_"
        f"{args.resolution:g}m_openeo_optimized.tif"
    )
    webp_path = (
        args.output_dir
        / f"sentinel2_l2a_true_color_B4_B3_B2_{date_label}_"
        "openeo_optimized.webp"
    )
    server_webp_path = (
        args.output_dir
        / f"sentinel2_l2a_true_color_B4_B3_B2_{date_label}_"
        "openeo_server.webp"
    )
    metadata_path = args.output_dir / "download_metadata.json"

    required_downloads = [imagery_path, aligned_dem_path, native_dem_path]
    if all(completed_raster(path) for path in required_downloads) and (
        server_webp_path.exists() and server_webp_path.stat().st_size > 0
    ):
        token = ""
        print("All source rasters exist; running local post-processing only.")
    else:
        client_id = os.environ.get("CDSE_CLIENT_ID")
        client_secret = os.environ.get("CDSE_CLIENT_SECRET")
        if not client_id or not client_secret:
            print(
                "Missing credentials. Set CDSE_CLIENT_ID and "
                "CDSE_CLIENT_SECRET, then run the command again.",
                file=sys.stderr,
            )
            return 2
        print("Requesting Copernicus Data Space access token...")
        token = get_access_token(client_id, client_secret)
    print("Downloading Sentinel-2 L2A imagery...")
    process_download(imagery_payload, imagery_path, token)
    print("Downloading pixel-aligned Copernicus DEM...")
    process_download(aligned_dem_payload, aligned_dem_path, token)
    print("Downloading approximate native-resolution Copernicus DEM...")
    process_download(native_dem_payload, native_dem_path, token)
    print("Downloading official openEO true-color WebP...")
    openeo_download(openeo_payload, server_webp_path, token)

    add_raster_metadata(imagery_path, aligned_dem_path, native_dem_path)
    grid_check = verify_matching_grid(imagery_path, aligned_dem_path)
    write_display_geotiff(imagery_path, display_path)
    write_rgb_preview(imagery_path, preview_path, webp_path)
    metadata = {
        **plan,
        "imagery_bands": ["B04", "B03", "B02"],
        "imagery_band_order": "RGB",
        "imagery_units": "DN for spectral bands",
        "dem_units": "metre",
        "dem_vertical_reference": "EGM2008 geoid",
        "grid_check": grid_check,
        "outputs": {
            "sentinel2_l2a_geotiff": str(imagery_path),
            "sentinel2_true_color_8bit_geotiff": str(display_path),
            "sentinel2_true_color_webp": str(webp_path),
            "sentinel2_true_color_openeo_server_webp": str(server_webp_path),
            "dem_aligned_geotiff": str(aligned_dem_path),
            "dem_native_geotiff": str(native_dem_path),
            "rgb_preview_png": str(preview_path),
            "metadata_json": str(metadata_path),
            "imagery_request_json": str(imagery_request_path),
            "aligned_dem_request_json": str(aligned_dem_request_path),
            "native_dem_request_json": str(native_dem_request_path),
            "openeo_true_color_request_json": str(openeo_request_path),
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
