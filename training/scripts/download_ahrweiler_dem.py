"""Download and crop the public Copernicus GLO-30 COG for the Ahrweiler AOI."""
import os

import rasterio
from rasterio.windows import from_bounds

URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_N50_00_E007_00_DEM/"
    "Copernicus_DSM_COG_10_N50_00_E007_00_DEM.tif"
)
OUT = "deliverables/ahrweiler_flood_2021/dem/Ahrweiler_Copernicus_GLO30_AOI.tif"
# west, south, east, north; WGS84
BOUNDS = (7.00, 50.45, 7.24, 50.58)

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_USE_HEAD"] = "NO"

with rasterio.open(URL) as source:
    window = from_bounds(*BOUNDS, transform=source.transform).round_offsets().round_lengths()
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        height=int(window.height),
        width=int(window.width),
        transform=source.window_transform(window),
        compress="deflate",
        tiled=True,
    )
    with rasterio.open(OUT, "w", **profile) as output:
        output.write(source.read(window=window))

print(OUT)
