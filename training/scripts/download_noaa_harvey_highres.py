from __future__ import annotations

import argparse
import io
import json
import math
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.fill import fillnodata
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds


WEB_MERCATOR_HALF_WORLD = 20_037_508.342789244
TILE_SIZE = 256
WMTS_CAPABILITIES_URL = (
    "https://storms.ngs.noaa.gov/storms/tilesd/services/"
    "tileserver.php/wmts/1.0.0/WMTSCapabilities.xml"
)
WMTS_CACHE = Path(
    r"D:\project\water_segment\data\raw"
    r"\noaa_harvey_wmts_capabilities.xml"
)
USGS_3DEP_QUERY_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/query"
)


@dataclass(frozen=True)
class ImageryChoice:
    layer: str
    zoom: int
    layer_bounds: tuple[float, float, float, float]
    tile_count: int
    selection_mode: str


@dataclass(frozen=True)
class DemChoice:
    source: str
    title: str
    nominal_resolution_m: float | None
    vertical_datum: str | None
    selection_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically download the highest available NOAA imagery and "
            "the finest available USGS 3DEP DEM for a WGS84 bounding box."
        )
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
    )
    parser.add_argument(
        "--layer",
        default="auto",
        help="NOAA WMTS layer name, or 'auto' (default).",
    )
    parser.add_argument(
        "--wmts-capabilities-url",
        default=WMTS_CAPABILITIES_URL,
        help=(
            "Event-specific NOAA WMTS capabilities URL. Defaults to "
            "Hurricane Harvey."
        ),
    )
    parser.add_argument(
        "--zoom",
        default="auto",
        help="WMTS zoom integer, or 'auto' (default).",
    )
    parser.add_argument(
        "--dem",
        default="auto",
        help="Local DEM path/URL, or 'auto' for USGS 3DEP (default).",
    )
    parser.add_argument(
        "--source-gsd-m",
        type=float,
        default=None,
        help="Official source imagery GSD in metres, when known.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=1500,
        help="Safety limit for one imagery download (default: 1500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select the best imagery and DEM, print the result, then exit.",
    )
    return parser.parse_args()


def request_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 60,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after retries: {url}") from last_error


def longitude_to_tile_x(longitude: float, zoom: int) -> float:
    return (longitude + 180.0) / 360.0 * (2**zoom)


def latitude_to_tile_y(latitude: float, zoom: int) -> float:
    latitude_radians = math.radians(latitude)
    return (
        1.0 - math.asinh(math.tan(latitude_radians)) / math.pi
    ) / 2.0 * (2**zoom)


def tile_resolution(zoom: int) -> float:
    return 2 * WEB_MERCATOR_HALF_WORLD / (TILE_SIZE * 2**zoom)


def tile_range(
    bbox: tuple[float, float, float, float], zoom: int
) -> tuple[int, int, int, int]:
    west, south, east, north = bbox
    x_min = math.floor(longitude_to_tile_x(west, zoom))
    x_max = math.floor(
        longitude_to_tile_x(math.nextafter(east, -math.inf), zoom)
    )
    y_min = math.floor(latitude_to_tile_y(north, zoom))
    y_max = math.floor(
        latitude_to_tile_y(math.nextafter(south, math.inf), zoom)
    )
    return x_min, x_max, y_min, y_max


def tile_count(
    bbox: tuple[float, float, float, float], zoom: int
) -> int:
    x_min, x_max, y_min, y_max = tile_range(bbox, zoom)
    return (x_max - x_min + 1) * (y_max - y_min + 1)


def load_wmts_capabilities(capabilities_url: str) -> bytes:
    try:
        content = request_with_retry(capabilities_url).content
        if capabilities_url == WMTS_CAPABILITIES_URL:
            WMTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            WMTS_CACHE.write_bytes(content)
        return content
    except RuntimeError:
        if capabilities_url == WMTS_CAPABILITIES_URL and WMTS_CACHE.exists():
            return WMTS_CACHE.read_bytes()
        raise


def parse_wmts_capabilities(
    content: bytes,
) -> tuple[dict[str, tuple[float, float, float, float]], int]:
    root = ET.fromstring(content)
    namespaces = {
        "wmts": "http://www.opengis.net/wmts/1.0",
        "ows": "http://www.opengis.net/ows/1.1",
    }
    layers: dict[str, tuple[float, float, float, float]] = {}
    for layer in root.findall(".//wmts:Contents/wmts:Layer", namespaces):
        identifier = layer.findtext("ows:Identifier", namespaces=namespaces)
        box = layer.find("ows:WGS84BoundingBox", namespaces)
        if not identifier or box is None:
            continue
        lower = box.findtext("ows:LowerCorner", namespaces=namespaces)
        upper = box.findtext("ows:UpperCorner", namespaces=namespaces)
        if not lower or not upper:
            continue
        west, south = map(float, lower.split())
        east, north = map(float, upper.split())
        layers[identifier] = (west, south, east, north)

    zooms: list[int] = []
    for identifier in root.findall(
        ".//wmts:TileMatrixSet/wmts:TileMatrix/ows:Identifier", namespaces
    ):
        if identifier.text:
            try:
                zooms.append(int(identifier.text))
            except ValueError:
                continue
    if not layers or not zooms:
        raise RuntimeError("NOAA WMTS capabilities contain no usable layers.")
    return layers, max(zooms)


def bbox_contains(
    container: tuple[float, float, float, float],
    requested: tuple[float, float, float, float],
) -> bool:
    return (
        container[0] <= requested[0]
        and container[1] <= requested[1]
        and container[2] >= requested[2]
        and container[3] >= requested[3]
    )


def bbox_intersects(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def download_tile(
    layer: str,
    zoom: int,
    x: int,
    y: int,
) -> tuple[int, int, np.ndarray | None]:
    url = f"https://stormscdn.ngs.noaa.gov/{layer}/{zoom}/{x}/{y}"
    response = request_with_retry(url)
    image = Image.open(io.BytesIO(response.content))
    if image.size == (1, 1):
        return x, y, None
    return x, y, np.asarray(image.convert("RGB"), dtype=np.uint8)


def tile_exists(layer: str, zoom: int, longitude: float, latitude: float) -> bool:
    x = math.floor(longitude_to_tile_x(longitude, zoom))
    y = math.floor(latitude_to_tile_y(latitude, zoom))
    return download_tile(layer, zoom, x, y)[2] is not None


def layer_has_bbox_coverage(
    layer: str,
    zoom: int,
    bbox: tuple[float, float, float, float],
) -> bool:
    west, south, east, north = bbox
    margin_x = min((east - west) * 0.02, 1e-5)
    margin_y = min((north - south) * 0.02, 1e-5)
    points = [
        ((west + east) / 2, (south + north) / 2),
        (west + margin_x, south + margin_y),
        (west + margin_x, north - margin_y),
        (east - margin_x, south + margin_y),
        (east - margin_x, north - margin_y),
    ]
    return all(tile_exists(layer, zoom, lon, lat) for lon, lat in points)


def choose_imagery(
    bbox: tuple[float, float, float, float],
    requested_layer: str,
    requested_zoom: str,
    max_tiles: int,
    capabilities_url: str,
) -> ImageryChoice:
    layers, service_max_zoom = parse_wmts_capabilities(
        load_wmts_capabilities(capabilities_url)
    )
    if requested_layer != "auto":
        if requested_layer not in layers:
            raise ValueError(f"WMTS layer does not exist: {requested_layer}")
        candidates = [requested_layer]
        mode = "manual layer"
    else:
        contained = [
            name
            for name, bounds in layers.items()
            if "rgb" in name.lower() and bbox_contains(bounds, bbox)
        ]
        candidates = contained or [
            name
            for name, bounds in layers.items()
            if "rgb" in name.lower() and bbox_intersects(bounds, bbox)
        ]
        candidates.sort(reverse=True)
        mode = "automatic highest available imagery"
    if not candidates:
        raise RuntimeError("No NOAA WMTS layer intersects the requested bbox.")

    if requested_zoom == "auto":
        zooms = range(service_max_zoom, -1, -1)
    else:
        try:
            zooms = [int(requested_zoom)]
        except ValueError as error:
            raise ValueError("--zoom must be an integer or 'auto'.") from error

    oversized: list[tuple[int, int]] = []
    for zoom in zooms:
        count = tile_count(bbox, zoom)
        if count > max_tiles:
            oversized.append((zoom, count))
            continue
        for layer in candidates:
            if layer_has_bbox_coverage(layer, zoom, bbox):
                return ImageryChoice(
                    layer=layer,
                    zoom=zoom,
                    layer_bounds=layers[layer],
                    tile_count=count,
                    selection_mode=mode,
                )
    if oversized:
        highest_zoom, count = oversized[0]
        raise RuntimeError(
            f"The highest valid search starts at zoom {highest_zoom}, but "
            f"the bbox needs {count} tiles (limit: {max_tiles}). Increase "
            "--max-tiles or use a smaller bbox."
        )
    raise RuntimeError(
        "No non-empty NOAA imagery tiles cover the full requested bbox."
    )


def query_usgs_dem_candidates(
    bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    west, south, east, north = bbox
    center = ((west + east) / 2, (south + north) / 2)
    response = request_with_retry(
        USGS_3DEP_QUERY_URL,
        params={
            "f": "json",
            "where": "1=1",
            "geometry": f"{center[0]},{center[1]}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": (
                "Name,title,URL,LowPS,Best,VerticalDatum,"
                "Resolution_X,Resolution_Y"
            ),
            "returnGeometry": "false",
        },
    )
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"USGS 3DEP query failed: {payload['error']}")
    candidates = [
        feature.get("attributes", {})
        for feature in payload.get("features", [])
        if feature.get("attributes", {}).get("URL")
    ]

    def sort_key(candidate: dict[str, Any]) -> tuple[float, float]:
        resolution = candidate.get("LowPS")
        try:
            resolution_value = float(resolution)
        except (TypeError, ValueError):
            resolution_value = math.inf
        best = candidate.get("Best")
        try:
            best_value = float(best)
        except (TypeError, ValueError):
            best_value = 0.0
        return resolution_value, -best_value

    return sorted(candidates, key=sort_key)


def source_covers_bbox(
    source: str,
    bbox: tuple[float, float, float, float],
) -> bool:
    try:
        with rasterio.open(source) as dataset:
            requested = transform_bounds(
                "EPSG:4326", dataset.crs, *bbox, densify_pts=21
            )
            tolerance_x = abs(dataset.transform.a) * 2
            tolerance_y = abs(dataset.transform.e) * 2
            return (
                dataset.bounds.left - tolerance_x <= requested[0]
                and dataset.bounds.bottom - tolerance_y <= requested[1]
                and dataset.bounds.right + tolerance_x >= requested[2]
                and dataset.bounds.top + tolerance_y >= requested[3]
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def choose_dem(
    bbox: tuple[float, float, float, float], requested_dem: str
) -> DemChoice:
    if requested_dem != "auto":
        source = str(Path(requested_dem).expanduser())
        if not Path(source).exists() and not source.startswith(("http://", "https://")):
            raise FileNotFoundError(f"DEM does not exist: {source}")
        if not source_covers_bbox(source, bbox):
            raise RuntimeError("The specified DEM does not fully cover the bbox.")
        with rasterio.open(source) as dataset:
            resolution = min(abs(dataset.res[0]), abs(dataset.res[1]))
        return DemChoice(
            source=source,
            title=Path(source).name,
            nominal_resolution_m=float(resolution),
            vertical_datum=None,
            selection_mode="manual DEM",
        )

    for candidate in query_usgs_dem_candidates(bbox):
        source = str(candidate["URL"])
        if not source_covers_bbox(source, bbox):
            continue
        try:
            nominal_resolution = float(candidate.get("LowPS"))
        except (TypeError, ValueError):
            nominal_resolution = None
        return DemChoice(
            source=source,
            title=str(candidate.get("title") or candidate.get("Name") or "USGS 3DEP"),
            nominal_resolution_m=nominal_resolution,
            vertical_datum=candidate.get("VerticalDatum"),
            selection_mode="automatic finest USGS 3DEP source",
        )
    raise RuntimeError("No USGS 3DEP DEM source fully covers the requested bbox.")


def download_mosaic(
    choice: ImageryChoice,
    bbox: tuple[float, float, float, float],
    workers: int,
) -> tuple[np.ndarray, Affine, list[list[int]], list[list[int]]]:
    x_min, x_max, y_min, y_max = tile_range(bbox, choice.zoom)
    coordinates = [
        (x, y)
        for y in range(y_min, y_max + 1)
        for x in range(x_min, x_max + 1)
    ]
    rows = y_max - y_min + 1
    columns = x_max - x_min + 1
    mosaic = np.zeros(
        (rows * TILE_SIZE, columns * TILE_SIZE, 3), dtype=np.uint8
    )
    missing_tiles: list[list[int]] = []
    fallback_tiles: list[list[int]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_tile, choice.layer, choice.zoom, x, y
            ): (x, y)
            for x, y in coordinates
        }
        for future in as_completed(futures):
            x, y, tile = future.result()
            if tile is None:
                _, _, parent = download_tile(
                    choice.layer, choice.zoom - 1, x // 2, y // 2
                )
                if parent is None:
                    missing_tiles.append([x, y])
                    continue
                quadrant = parent[
                    (y % 2) * 128 : (y % 2 + 1) * 128,
                    (x % 2) * 128 : (x % 2 + 1) * 128,
                ]
                tile = np.asarray(
                    Image.fromarray(quadrant).resize(
                        (TILE_SIZE, TILE_SIZE), Image.Resampling.BICUBIC
                    ),
                    dtype=np.uint8,
                )
                fallback_tiles.append([x, y])
            row_offset = (y - y_min) * TILE_SIZE
            column_offset = (x - x_min) * TILE_SIZE
            mosaic[
                row_offset : row_offset + TILE_SIZE,
                column_offset : column_offset + TILE_SIZE,
            ] = tile

    resolution = tile_resolution(choice.zoom)
    mosaic_left = -WEB_MERCATOR_HALF_WORLD + x_min * TILE_SIZE * resolution
    mosaic_top = WEB_MERCATOR_HALF_WORLD - y_min * TILE_SIZE * resolution
    mosaic_transform = Affine(
        resolution, 0, mosaic_left, 0, -resolution, mosaic_top
    )
    requested_3857 = transform_bounds(
        "EPSG:4326", "EPSG:3857", *bbox, densify_pts=21
    )
    crop_window = from_bounds(
        *requested_3857, transform=mosaic_transform
    ).round_offsets().round_lengths()
    crop_window = crop_window.intersection(
        Window(0, 0, mosaic.shape[1], mosaic.shape[0])
    )
    row_start = int(crop_window.row_off)
    row_end = row_start + int(crop_window.height)
    column_start = int(crop_window.col_off)
    column_end = column_start + int(crop_window.width)
    cropped = mosaic[row_start:row_end, column_start:column_end]
    cropped_transform = rasterio.windows.transform(
        crop_window, mosaic_transform
    )
    return cropped, cropped_transform, missing_tiles, fallback_tiles


def write_rgb_geotiff(
    path: Path,
    rgb: np.ndarray,
    transform: Affine,
    choice: ImageryChoice,
    output_ground_pixel_size_m: float,
    source_gsd_m: float | None,
) -> None:
    profile = {
        "driver": "GTiff",
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "deflate",
        "photometric": "RGB",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as output:
        output.write(np.moveaxis(rgb, 2, 0))
        output.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
        )
        output.set_band_description(1, "Red")
        output.set_band_description(2, "Green")
        output.set_band_description(3, "Blue")
        tags = {
            "source": "NOAA NGS Emergency Response Imagery",
            "acquisition_date": choice.layer[:8],
            "wmts_layer": choice.layer,
            "wmts_zoom": str(choice.zoom),
            "output_ground_pixel_size_m": (
                f"{output_ground_pixel_size_m:.6f}"
            ),
            "selection_mode": choice.selection_mode,
        }
        if source_gsd_m is not None:
            tags["nominal_source_ground_sample_distance_m"] = str(
                source_gsd_m
            )
        output.update_tags(**tags)


def clip_native_dem(
    choice: DemChoice,
    bbox: tuple[float, float, float, float],
    output_path: Path,
) -> dict[str, Any]:
    with rasterio.open(choice.source) as source:
        requested = transform_bounds(
            "EPSG:4326", source.crs, *bbox, densify_pts=21
        )
        window = from_bounds(*requested, transform=source.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, source.width, source.height))
        data = source.read(1, window=window).astype(np.float32)
        source_nodata = source.nodata
        if source_nodata is not None:
            data[data == source_nodata] = np.nan
        data[~np.isfinite(data)] = np.nan
        transform = source.window_transform(window)
        profile = {
            "driver": "GTiff",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "dtype": "float32",
            "crs": source.crs,
            "transform": transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
        }
        crs_name = source.crs.to_string()
        native_resolution = [float(abs(source.res[0])), float(abs(source.res[1]))]
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(data, 1)
        output.set_band_description(1, "Bare-earth elevation")
        output.update_tags(
            source=choice.source,
            source_title=choice.title,
            vertical_reference=choice.vertical_datum or "not reported",
            selection_mode=choice.selection_mode,
        )
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        raise RuntimeError("The selected DEM contains no valid elevations.")
    return {
        "source": choice.source,
        "source_title": choice.title,
        "selection_mode": choice.selection_mode,
        "catalog_nominal_resolution_m": choice.nominal_resolution_m,
        "native_resolution": native_resolution,
        "native_crs": crs_name,
        "vertical_datum": choice.vertical_datum,
        "native_width": int(data.shape[1]),
        "native_height": int(data.shape[0]),
        "elevation_min": float(valid.min()),
        "elevation_max": float(valid.max()),
    }


def align_dem(
    source_path: Path,
    output_path: Path,
    width: int,
    height: int,
    transform: Affine,
    choice: DemChoice,
) -> dict[str, Any]:
    destination = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(source_path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs="EPSG:3857",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    valid_mask = np.isfinite(destination)
    if valid_mask.any() and not valid_mask.all():
        destination = fillnodata(
            destination,
            mask=valid_mask.astype(np.uint8),
            max_search_distance=10,
        ).astype(np.float32)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:3857",
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
    }
    with rasterio.open(output_path, "w", **profile) as output:
        output.write(destination, 1)
        output.set_band_description(1, "Bare-earth elevation")
        output.update_tags(
            source=choice.source,
            source_title=choice.title,
            purpose="Pixel-aligned to NOAA aerial imagery",
            interpolation="bilinear",
            native_resolution_m=str(choice.nominal_resolution_m),
            vertical_reference=choice.vertical_datum or "not reported",
        )
    valid = destination[np.isfinite(destination)]
    if valid.size == 0:
        raise RuntimeError("Aligned DEM contains no valid elevations.")
    return {
        "valid_pixels": int(valid.size),
        "total_pixels": int(destination.size),
        "elevation_min": float(valid.min()),
        "elevation_max": float(valid.max()),
    }


def write_preview(rgb: np.ndarray, path: Path) -> None:
    image = Image.fromarray(rgb, mode="RGB")
    image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    image.save(path)


def main() -> None:
    args = parse_args()
    bbox = tuple(args.bbox)
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -85 <= south < north <= 85):
        raise ValueError("Invalid WGS84 bbox.")
    if args.max_tiles < 1:
        raise ValueError("--max-tiles must be positive.")

    imagery_choice = choose_imagery(
        bbox,
        args.layer,
        args.zoom,
        args.max_tiles,
        args.wmts_capabilities_url,
    )
    dem_choice = choose_dem(bbox, args.dem)
    selection = {
        "imagery": {
            "layer": imagery_choice.layer,
            "acquisition_date": imagery_choice.layer[:8],
            "zoom": imagery_choice.zoom,
            "tile_count": imagery_choice.tile_count,
            "selection_mode": imagery_choice.selection_mode,
            "wmts_capabilities_url": args.wmts_capabilities_url,
        },
        "dem": {
            "source": dem_choice.source,
            "title": dem_choice.title,
            "nominal_resolution_m": dem_choice.nominal_resolution_m,
            "vertical_datum": dem_choice.vertical_datum,
            "selection_mode": dem_choice.selection_mode,
        },
    }
    print("Selected highest available sources:")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb, rgb_transform, missing_tiles, fallback_tiles = download_mosaic(
        imagery_choice, bbox, args.workers
    )
    if missing_tiles:
        raise RuntimeError(
            f"{len(missing_tiles)} imagery tiles were empty. Use --dry-run "
            "to inspect the selection or specify a lower --zoom."
        )

    resolution = tile_resolution(imagery_choice.zoom)
    ground_pixel_size_m = resolution * math.cos(
        math.radians((south + north) / 2)
    )
    prefix = (
        f"noaa_{imagery_choice.layer}_z{imagery_choice.zoom}"
    )
    imagery_path = args.output_dir / f"{prefix}_rgb.tif"
    preview_path = args.output_dir / f"{prefix}_preview.png"
    native_dem_path = args.output_dir / "usgs_3dep_dem_native.tif"
    aligned_dem_path = (
        args.output_dir
        / f"usgs_3dep_dem_aligned_to_{prefix}.tif"
    )
    metadata_path = args.output_dir / "metadata.json"

    write_rgb_geotiff(
        imagery_path,
        rgb,
        rgb_transform,
        imagery_choice,
        ground_pixel_size_m,
        args.source_gsd_m,
    )
    write_preview(rgb, preview_path)
    native_dem_stats = clip_native_dem(
        dem_choice, bbox, native_dem_path
    )
    aligned_dem_stats = align_dem(
        native_dem_path,
        aligned_dem_path,
        rgb.shape[1],
        rgb.shape[0],
        rgb_transform,
        dem_choice,
    )

    actual_bounds_3857 = rasterio.transform.array_bounds(
        rgb.shape[0], rgb.shape[1], rgb_transform
    )
    actual_bounds_wgs84 = transform_bounds(
        "EPSG:3857",
        "EPSG:4326",
        *actual_bounds_3857,
        densify_pts=21,
    )
    metadata = {
        **selection,
        "requested_bounds_wgs84": {
            "west": west,
            "south": south,
            "east": east,
            "north": north,
        },
        "imagery": {
            **selection["imagery"],
            "nominal_source_ground_sample_distance_m": args.source_gsd_m,
            "output_ground_pixel_size_m": ground_pixel_size_m,
            "crs": "EPSG:3857",
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "actual_bounds_wgs84": {
                "left": actual_bounds_wgs84[0],
                "bottom": actual_bounds_wgs84[1],
                "right": actual_bounds_wgs84[2],
                "top": actual_bounds_wgs84[3],
            },
            "missing_tile_count": 0,
            "lower_zoom_fallback_tile_count": len(fallback_tiles),
            "lower_zoom_fallback_tiles": fallback_tiles,
            "black_pixel_count": int(np.all(rgb == 0, axis=2).sum()),
        },
        "dem": {
            **selection["dem"],
            "native": native_dem_stats,
            "aligned": aligned_dem_stats,
        },
        "outputs": {
            "rgb_geotiff": str(imagery_path),
            "preview_png": str(preview_path),
            "native_dem_geotiff": str(native_dem_path),
            "aligned_dem_geotiff": str(aligned_dem_path),
            "metadata_json": str(metadata_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
