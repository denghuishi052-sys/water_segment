"""Rebuild the raster basemap embedded in the EMSR517 GeoPDF, without vector overlays."""
from io import BytesIO
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

pdf = Path("deliverables/ahrweiler_flood_2021/source/EMSR517_AOI15_20210720_pleiades_0p5m_map.pdf")
out = Path("deliverables/ahrweiler_flood_2021/imagery/Ahrweiler_unannotated_EMS_raster_basemap.jpg")

images = PdfReader(pdf).pages[0].images
# The GeoPDF stores its 4394-pixel-wide raster background as vertical JPEG strips.
strips = [item.image.convert("RGB") for item in images if item.image.width == 4394]
height = sum(image.height for image in strips)
canvas = Image.new("RGB", (4394, height))
y = 0
for strip in strips:
    canvas.paste(strip, (0, y))
    y += strip.height

canvas.save(out, "JPEG", quality=95, subsampling=0)
print(f"Wrote {out} ({canvas.width} x {canvas.height})")
