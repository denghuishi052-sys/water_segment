import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "GEHistoricalImagery" / "app" / "GEHistoricalImagery.bat"
BASE = ROOT / "data" / "processed" / "google_earth_flood_samples_global"

SAMPLES = [
    {
        "id": "houston_harvey_addicks_2017",
        "event": "Hurricane Harvey flooding, Houston / Addicks Reservoir area, Texas, USA",
        "date": "2017/08/31",
        "zoom": 20,
        "bounds": {"south": 29.7570, "north": 29.7670, "west": -95.6820, "east": -95.6620},
        "note": "Urban/reservoir-edge flooding after Hurricane Harvey.",
    },
    {
        "id": "bangkok_thailand_flood_2011",
        "event": "2011 Thailand floods, northern Bangkok / Don Mueang-Rangsit area",
        "date": "2011/11/05",
        "zoom": 19,
        "bounds": {"south": 13.9220, "north": 13.9320, "west": 100.5970, "east": 100.6170},
        "note": "Bangkok floodwater during the late-October to November 2011 flood period.",
    },
    {
        "id": "midland_michigan_dam_flood_2020",
        "event": "May 2020 Midland, Michigan flood after Edenville and Sanford dam failures",
        "date": "2020/05/22",
        "zoom": 19,
        "bounds": {"south": 43.6200, "north": 43.6300, "west": -84.2700, "east": -84.2500},
        "note": "Tittabawassee River flood impacts around Midland shortly after dam failures.",
    },
    {
        "id": "brisbane_australia_flood_2022",
        "event": "February-March 2022 Brisbane flood, Queensland, Australia",
        "date": "2022/03/06",
        "zoom": 20,
        "bounds": {"south": -27.4750, "north": -27.4650, "west": 153.0100, "east": 153.0300},
        "note": "Brisbane River flood aftermath / high-water urban river corridor sample.",
    },
    {
        "id": "chennai_india_flood_2015",
        "event": "December 2015 Chennai floods, Tamil Nadu, India",
        "date": "2015/12/05",
        "zoom": 19,
        "bounds": {"south": 12.9950, "north": 13.0050, "west": 80.1700, "east": 80.1900},
        "note": "Urban flood sample near the Adyar/Chennai airport flood-affected corridor.",
    },
]


def run(cmd: list[str], timeout: int = 360) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True,
        timeout=timeout,
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
            "event": sample["event"],
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
            "zoom": sample["zoom"],
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
        date_slug = sample["date"].replace("/", "_")
        tif_path = out_dir / f"{sample['id']}_{date_slug}_z{sample['zoom']}_exact.tif"
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
                str(sample["zoom"]),
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
