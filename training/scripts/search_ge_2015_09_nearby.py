import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "GEHistoricalImagery" / "app" / "GEHistoricalImagery.bat"
OUT = ROOT / "data" / "processed" / "google_earth_36.05_36.06_139.57_139.59"


def run_info(lat: float, lon: float) -> str:
    proc = subprocess.run(
        [
            str(EXE),
            "info",
            "--location",
            f"{lat},{lon}",
            "--min-zoom",
            "18",
            "--max-zoom",
            "19",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="ignore",
        capture_output=True,
        timeout=45,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lats = [35.98, 36.00, 36.02, 36.04, 36.055, 36.07, 36.09, 36.11, 36.13]
    lons = [139.48, 139.51, 139.54, 139.57, 139.58, 139.59, 139.62, 139.65, 139.68]
    hits = []
    for lat in lats:
        for lon in lons:
            text = run_info(lat, lon)
            dates = sorted(set(re.findall(r"imagery_date = (2015/09/\d{2})", text)))
            if dates:
                hits.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "dates": dates,
                        "distance_deg_from_requested_center": math.hypot(
                            lat - 36.055, lon - 139.58
                        ),
                    }
                )
                print(f"HIT {lat},{lon}: {', '.join(dates)}")

    hits.sort(key=lambda item: item["distance_deg_from_requested_center"])
    (OUT / "nearby_2015_09_hits.json").write_text(
        json.dumps(hits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(hits[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
