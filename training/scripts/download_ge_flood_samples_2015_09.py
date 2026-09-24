import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "GEHistoricalImagery" / "app" / "GEHistoricalImagery.bat"
BASE = ROOT / "data" / "processed" / "google_earth_flood_samples_2015_09"

SAMPLES = [
    {
        "id": "joso_kinu_flood_center",
        "date": "2015/09/11",
        "bounds": {"south": 36.0985, "north": 36.1085, "west": 139.9822, "east": 140.0022},
        "note": "Kinu River / Joso flood, inundated fields and settlement edge.",
    },
    {
        "id": "joso_kinu_flood_south",
        "date": "2015/09/11",
        "bounds": {"south": 36.0885, "north": 36.0985, "west": 139.9822, "east": 140.0022},
        "note": "Adjacent southern inundation sample near Joso.",
    },
    {
        "id": "joso_kinu_flood_east",
        "date": "2015/09/11",
        "bounds": {"south": 36.0985, "north": 36.1085, "west": 140.0022, "east": 140.0222},
        "note": "Adjacent eastern inundation/river-edge sample near Joso.",
    },
    {
        "id": "joso_kinu_flood_north",
        "date": "2015/09/11",
        "bounds": {"south": 36.1085, "north": 36.1185, "west": 139.9822, "east": 140.0022},
        "note": "Adjacent northern inundation sample near Joso.",
    },
]


def run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True,
        timeout=180,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(text)
    return text


def make_preview_and_metadata(sample: dict, tif_path: Path) -> dict:
    preview_path = tif_path.with_name(tif_path.stem + "_preview_1600px.png")
    metadata_path = tif_path.with_name("metadata.json")
    with rasterio.open(tif_path) as src:
        arr = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        image = Image.fromarray(arr)
        image.thumbnail((1600, 1600))
        image.save(preview_path)
        bounds = src.bounds
        metadata = {
            "source": "GEHistoricalImagery / Google Earth Time Machine",
            "event": "September 2015 Kanto-Tohoku heavy rainfall / Kinu River flooding, Joso, Ibaraki, Japan",
            "sample_id": sample["id"],
            "date": sample["date"],
            "note": sample["note"],
            "requested_bounds_wgs84": sample["bounds"],
            "actual_bounds_wgs84": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes,
            "resolution_degrees": src.res,
            "zoom": 19,
            "exact_date": True,
            "outputs": {
                "geotiff": str(tif_path),
                "world_file": str(tif_path.with_suffix(".tfw")),
                "preview_png": str(preview_path),
            },
        }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    all_metadata = []
    for sample in SAMPLES:
        out_dir = BASE / sample["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        tif_path = out_dir / f"{sample['id']}_2015_09_11_z19_exact.tif"
        b = sample["bounds"]
        if not tif_path.exists():
            cmd = [
                str(EXE),
                "download",
                "--lower-left",
                f"{b['south']},{b['west']}",
                "--upper-right",
                f"{b['north']},{b['east']}",
                "--zoom",
                "19",
                "--date",
                sample["date"],
                "--exact-date",
                "--output",
                str(tif_path.relative_to(ROOT)),
            ]
            print(f"Downloading {sample['id']} ...")
            run(cmd)
        metadata = make_preview_and_metadata(sample, tif_path)
        all_metadata.append(metadata)
        print(f"OK {sample['id']}: {metadata['width']}x{metadata['height']}")

    (BASE / "samples_index.json").write_text(
        json.dumps(all_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
