# Copernicus imagery and DEM downloader

`download_cdse_imagery_dem.py` downloads matching products from the official
Copernicus Data Space Sentinel Hub Process API:

- Sentinel-2 L2A true-color GeoTIFF containing only B04, B03, and B02 in
  red-green-blue order.
- Display-ready 8-bit RGB GeoTIFF and WebP generated with the Copernicus
  openEO `true_color` parameters: `maxR=3`, `midR=0.13`, `sat=1.2`,
  `gamma=1.8`, and `gOff=0.01`.
- Copernicus GLO-30 DEM aligned pixel-for-pixel to the imagery.
- Copernicus GLO-30 DEM at approximately 30 m native resolution.
- RGB preview, metadata, and reproducible request JSON files.

## Authentication

Create an OAuth client in the Copernicus Data Space Sentinel Hub dashboard,
then set the credentials in the current PowerShell session:

```powershell
$env:CDSE_CLIENT_ID = "your-client-id"
$env:CDSE_CLIENT_SECRET = "your-client-secret"
```

The script reads these variables at runtime and does not save the secret.

The highest native spatial resolution of open Sentinel-2 imagery is 10 m.
The script rejects `--resolution` values below 10 m because those values would
only enlarge the output through interpolation, without adding spatial detail.

## Example

Download the Houston Hurricane Harvey area for 2017-08-31:

```powershell
python D:\project\water_segment\scripts\download_cdse_imagery_dem.py `
  --bbox -95.682 29.757 -95.662 29.767 `
  --start-date 2017-08-31 `
  --end-date 2017-09-05 `
  --resolution 10 `
  --max-cloud 80 `
  --output-dir D:\project\water_segment\data\processed\cdse_houston_2017
```

Use a short date range when the acquisition date matters. The API chooses the
least-cloudy Sentinel-2 tile within that range.

## Dry run

Generate and inspect API request JSON files without credentials or downloads:

```powershell
python D:\project\water_segment\scripts\download_cdse_imagery_dem.py `
  --bbox -95.682 29.757 -95.662 29.767 `
  --start-date 2017-08-31 `
  --end-date 2017-09-05 `
  --output-dir D:\project\water_segment\data\processed\cdse_houston_2017 `
  --dry-run
```

## Output interpretation

The spectral bands are stored as Sentinel-2 L2A digital numbers. The DEM band
contains elevation in metres relative to the EGM2008 geoid. The aligned DEM is
resampled to the imagery grid for pixel-wise use; this does not increase the
native terrain detail beyond approximately 30 m.

Use the 16-bit file for spectral analysis and the `_8bit_display.tif` file for
direct viewing in ordinary image software.
