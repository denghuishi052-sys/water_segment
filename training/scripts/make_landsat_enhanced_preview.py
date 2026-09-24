from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image


INPUT = Path("data/processed/lat36.1035_lon139.9922_2015_09/image_multiband_georef.tif")
OUTPUT_DIR = Path("data/processed/lat36.1035_lon139.9922_2015_09")


def stretch(values: np.ndarray) -> np.ndarray:
    arr = values.astype("float32")
    arr[arr == 0] = np.nan
    lo, hi = np.nanpercentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1), 0, 1)
    arr[~np.isfinite(arr)] = 0
    return (arr * 255).astype("uint8")


def main() -> None:
    with rasterio.open(INPUT) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        profile = src.profile.copy()
        transform = src.transform * src.transform.scale(0.5, 0.5)

    rgb = np.dstack([stretch(red), stretch(green), stretch(blue)])
    upscaled = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    enhanced = cv2.detailEnhance(enhanced, sigma_s=8, sigma_r=0.12)

    png_path = OUTPUT_DIR / "preview_rgb_enhanced_15m_display.png"
    Image.fromarray(enhanced).save(png_path)

    profile.update(
        height=enhanced.shape[0],
        width=enhanced.shape[1],
        count=3,
        dtype="uint8",
        transform=transform,
        compress="deflate",
        tiled=True,
    )
    tif_path = OUTPUT_DIR / "image_rgb_enhanced_15m_display_georef.tif"
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(np.moveaxis(enhanced, -1, 0))
        dst.set_band_description(1, "red_enhanced_display")
        dst.set_band_description(2, "green_enhanced_display")
        dst.set_band_description(3, "blue_enhanced_display")
        dst.update_tags(
            note="Display-only 2x cubic upsample from 30m Landsat SR; not true 15m source resolution."
        )

    print(png_path)
    print(tif_path)


if __name__ == "__main__":
    main()
