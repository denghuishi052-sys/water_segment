import json
from pathlib import Path

import numpy as np
import planetary_computer
import rasterio
from PIL import Image
from pyproj import Transformer
from pystac_client import Client
from rasterio.windows import from_bounds


LON = 139.9922
LAT = 36.1035
DATE_RANGE = "2015-09-01/2015-09-30"
OUTPUT_DIR = Path("data/processed/lat36.1035_lon139.9922_2015_09")
HALF_SIZE_M = 2500


def stretch_to_byte(values: np.ndarray, nodata: float | int | None = None) -> np.ndarray:
    arr = values.astype("float32")
    if nodata is not None:
        arr[arr == nodata] = np.nan
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros(arr.shape, dtype="uint8")

    lo, hi = np.nanpercentile(arr, [2, 98])
    if hi <= lo:
        hi = lo + 1
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    arr[~valid] = 0
    return (arr * 255).astype("uint8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        intersects={"type": "Point", "coordinates": [LON, LAT]},
        datetime=DATE_RANGE,
        max_items=20,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No Landsat items found for the requested point and month.")

    items.sort(
        key=lambda item: (
            0 if item.properties.get("landsat:collection_category") == "T1" else 1,
            item.properties.get("eo:cloud_cover", 999),
        )
    )
    item = planetary_computer.sign(items[0])
    item_dict = item.to_dict()
    proj_code = item.properties.get("proj:code")
    epsg = item.properties.get("proj:epsg")
    if epsg is None and isinstance(proj_code, str) and proj_code.upper().startswith("EPSG:"):
        epsg = int(proj_code.split(":", 1)[1])
    if epsg is None:
        raise RuntimeError("Selected STAC item does not include EPSG projection metadata.")

    transformer_to_item = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    center_x, center_y = transformer_to_item.transform(LON, LAT)
    left = center_x - HALF_SIZE_M
    right = center_x + HALF_SIZE_M
    bottom = center_y - HALF_SIZE_M
    top = center_y + HALF_SIZE_M

    bands = [
        ("blue", "SR_B2"),
        ("green", "SR_B3"),
        ("red", "SR_B4"),
        ("nir08", "SR_B5"),
        ("swir16", "SR_B6"),
        ("swir22", "SR_B7"),
        ("qa_pixel", "QA_PIXEL"),
    ]

    with rasterio.open(item.assets["red"].href) as src:
        window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
        transform = src.window_transform(window)
        profile = src.profile.copy()
        height = int(window.height)
        width = int(window.width)

    arrays = []
    band_descriptions = []
    for asset_key, description in bands:
        with rasterio.open(item.assets[asset_key].href) as src:
            arrays.append(src.read(1, window=window))
            band_descriptions.append(description)

    stack = np.stack(arrays)
    profile.update(
        driver="GTiff",
        height=height,
        width=width,
        count=len(bands),
        transform=transform,
        compress="deflate",
        tiled=True,
    )

    multiband_path = OUTPUT_DIR / "image_multiband_georef.tif"
    with rasterio.open(multiband_path, "w", **profile) as dst:
        dst.write(stack)
        for idx, description in enumerate(band_descriptions, start=1):
            dst.set_band_description(idx, description)
        dst.update_tags(
            source_item=item.id,
            center_lon=str(LON),
            center_lat=str(LAT),
            date_range=DATE_RANGE,
        )

    blue, green, red = arrays[0], arrays[1], arrays[2]
    rgb = np.dstack(
        [
            stretch_to_byte(red, nodata=0),
            stretch_to_byte(green, nodata=0),
            stretch_to_byte(blue, nodata=0),
        ]
    )

    preview_path = OUTPUT_DIR / "preview_rgb.png"
    Image.fromarray(rgb).save(preview_path)

    rgb_tif_profile = profile.copy()
    rgb_tif_profile.update(count=3, dtype="uint8")
    rgb_tif_path = OUTPUT_DIR / "image_rgb_georef.tif"
    with rasterio.open(rgb_tif_path, "w", **rgb_tif_profile) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))
        dst.set_band_description(1, "red_visual")
        dst.set_band_description(2, "green_visual")
        dst.set_band_description(3, "blue_visual")

    corners_item_crs = {
        "upper_left": [left, top],
        "upper_right": [right, top],
        "lower_right": [right, bottom],
        "lower_left": [left, bottom],
    }
    corners_wgs84 = {
        name: list(transformer_to_wgs84.transform(x, y))
        for name, (x, y) in corners_item_crs.items()
    }

    polygon = [
        corners_wgs84["upper_left"],
        corners_wgs84["upper_right"],
        corners_wgs84["lower_right"],
        corners_wgs84["lower_left"],
        corners_wgs84["upper_left"],
    ]
    bounds_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"source_item": item.id, "date_range": DATE_RANGE},
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
            }
        ],
    }

    metadata = {
        "source": "Microsoft Planetary Computer STAC landsat-c2-l2",
        "selected_item": item.id,
        "datetime": item.properties.get("datetime"),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "collection_category": item.properties.get("landsat:collection_category"),
        "platform": item.properties.get("platform"),
        "wrs_path": item.properties.get("landsat:wrs_path"),
        "wrs_row": item.properties.get("landsat:wrs_row"),
        "center_wgs84": {"lon": LON, "lat": LAT},
        "clip_size_m": HALF_SIZE_M * 2,
        "crs": f"EPSG:{epsg}",
        "pixel_size_m": [profile["transform"].a, abs(profile["transform"].e)],
        "width": width,
        "height": height,
        "transform": list(transform)[:6],
        "corners_wgs84": corners_wgs84,
        "assets": {key: item.assets[key].href for key, _ in bands},
        "outputs": {
            "multiband_geotiff": str(multiband_path),
            "rgb_geotiff": str(rgb_tif_path),
            "preview_png": str(preview_path),
        },
    }

    (OUTPUT_DIR / "selected_item_stac.json").write_text(
        json.dumps(item_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "bounds_wgs84.geojson").write_text(
        json.dumps(bounds_geojson, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
