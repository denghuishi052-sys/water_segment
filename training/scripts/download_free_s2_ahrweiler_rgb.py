"""Download a no-annotation, native-pixel Sentinel-2 RGB crop for the Ahr flood AOI."""
from pathlib import Path

import numpy as np
import planetary_computer
import pystac
import rasterio
import requests
from PIL import Image
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

ITEM_ID = "S2A_MSIL2A_20210721T104031_R008_T32ULA_20210721T223912"
# WGS84, around Bad Neuenahr-Ahrweiler / Ahr Valley
AOI = (7.00, 50.45, 7.24, 50.58)
OUT_DIR = Path("deliverables/ahrweiler_flood_2021/free_sentinel2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

response = requests.get(
    f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{ITEM_ID}",
    timeout=60,
)
response.raise_for_status()
item = planetary_computer.sign(pystac.Item.from_dict(response.json()))
acquisition_utc = str(item.properties["datetime"])
acquired = acquisition_utc[:10].replace("-", "")

datasets = [rasterio.open(item.assets[band].href) for band in ("B04", "B03", "B02")]
try:
    bounds = transform_bounds("EPSG:4326", datasets[0].crs, *AOI, densify_pts=21)
    window = from_bounds(*bounds, transform=datasets[0].transform).round_offsets().round_lengths()
    data = np.stack([dataset.read(1, window=window) for dataset in datasets])
    profile = datasets[0].profile.copy()
    profile.update(
        driver="GTiff", count=3, height=data.shape[1], width=data.shape[2],
        transform=datasets[0].window_transform(window), compress="deflate", tiled=True,
    )
finally:
    for dataset in datasets:
        dataset.close()

tif_path = OUT_DIR / f"Sentinel2_{acquired}_Ahrweiler_RGB_native_10m.tif"
with rasterio.open(tif_path, "w", **profile) as target:
    target.write(data)
    target.update_tags(
        source_item=ITEM_ID,
        acquisition_utc=acquisition_utc,
        bands="B04,B03,B02",
        processing="Sentinel-2 L2A surface reflectance; cropped only, no visual annotations",
    )

# This PNG is only a display rendition: each RGB channel is independently clipped at 2-98%.
rgb = np.empty(data.shape, dtype=np.uint8)
for index in range(3):
    lo, hi = np.percentile(data[index][data[index] > 0], (2, 98))
    rgb[index] = np.clip((data[index] - lo) * 255 / (hi - lo), 0, 255)
png_path = OUT_DIR / f"Sentinel2_{acquired}_Ahrweiler_RGB_native_10m_preview.png"
Image.fromarray(np.moveaxis(rgb, 0, -1)).save(png_path)

metadata_path = OUT_DIR / f"Sentinel2_{acquired}_Ahrweiler_RGB_native_10m_metadata.txt"
metadata_path.write_text(
    "Sentinel-2 L2A free source\n"
    f"STAC item: {ITEM_ID}\n"
    f"Acquisition UTC: {acquisition_utc}\n"
    "Native spatial resolution: 10 m\n"
    "AOI WGS84 (west,south,east,north): 7.00,50.45,7.24,50.58\n"
    "GeoTIFF bands: B04 (red), B03 (green), B02 (blue); no labels or map overlays.\n"
    "PNG is a contrast-stretched preview derived from the GeoTIFF.\n",
    encoding="utf-8",
)
print(tif_path)
print(png_path)
