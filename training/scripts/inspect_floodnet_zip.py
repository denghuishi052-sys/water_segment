#!/usr/bin/env python
"""Quick inspection of floodnet.zip to understand its structure."""
from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

ZIP_PATH = r"C:\Users\17473\Downloads\floodnet.zip"


def main():
    zip_path = Path(ZIP_PATH)
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found!")
        sys.exit(1)

    with ZipFile(zip_path) as z:
        names = z.namelist()
        print(f"Total entries: {len(names)}")
        print()

        # Show first 100 entries
        print("=== First 100 entries ===")
        for n in names[:100]:
            info = z.getinfo(n)
            size_mb = info.file_size / (1024 * 1024)
            print(f"  {n}  ({size_mb:.2f} MB)")

        # Find top-level directories
        top_dirs: dict[str, int] = {}
        for n in names:
            parts = n.split("/")
            if len(parts) >= 2:
                top = parts[0] + "/" + parts[1]
                top_dirs[top] = top_dirs.get(top, 0) + 1
            elif len(parts) == 1 and not n.endswith("/"):
                top_dirs[n] = top_dirs.get(n, 0) + 1

        print()
        print("=== Top-level directory structure (first 50) ===")
        for k, v in sorted(top_dirs.items())[:50]:
            print(f"  {k}/  ({v} files)")

        # Find image files
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        img_files = [n for n in names if Path(n).suffix.lower() in img_exts]
        print()
        print(f"=== Image files: {len(img_files)} ===")
        if img_files:
            print(f"  First: {img_files[0]}")
            print(f"  Last:  {img_files[-1]}")
            # Sample from different subdirs
            subdirs: dict[str, list[str]] = {}
            for f in img_files:
                parts = f.split("/")
                key = "/".join(parts[:-1]) if len(parts) > 1 else "(root)"
                subdirs.setdefault(key, []).append(f)
            print(f"  Image directories ({len(subdirs)}):")
            for k, v in sorted(subdirs.items()):
                print(f"    {k}/  ({len(v)} files)")

        # Find label/mask files
        mask_keywords = ["mask", "label", "ann", "gt", "seg", "truth"]
        label_files = [n for n in names if any(kw in n.lower() for kw in mask_keywords)]
        print()
        print(f"=== Potential label/mask files: {len(label_files)} ===")
        for f in label_files[:30]:
            info = z.getinfo(f)
            size_mb = info.file_size / (1024 * 1024)
            print(f"  {f}  ({size_mb:.2f} MB)")
        if len(label_files) > 30:
            print(f"  ... and {len(label_files) - 30} more")

        # Find CSV/TXT/YAML metadata files
        meta_exts = {".csv", ".txt", ".yaml", ".yml", ".json", ".xml"}
        meta_files = [n for n in names if Path(n).suffix.lower() in meta_exts]
        print()
        print(f"=== Metadata files: {len(meta_files)} ===")
        for f in meta_files[:20]:
            print(f"  {f}")
            # Try to show content for small files
            try:
                info = z.getinfo(f)
                if info.file_size < 2000:
                    with z.open(f) as fh:
                        content = fh.read().decode("utf-8", errors="replace")
                        print(f"    Content: {content[:500]}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
