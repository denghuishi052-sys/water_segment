"""Build a compact, balanced multi-domain water dataset using hard links."""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def linked_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def is_positive(image: Path, labels: Path) -> bool:
    label = labels / f"{image.stem}.txt"
    return label.is_file() and bool(label.read_text(encoding="utf-8").strip())


def add_samples(images: list[Path], labels: Path, split: str, source: str, output: Path) -> int:
    count = 0
    for image in images:
        prefix = f"{source}_{image.name}"
        linked_copy(image, output / "images" / split / prefix)
        label = labels / f"{image.stem}.txt"
        if label.is_file():
            linked_copy(label, output / "labels" / split / f"{source}_{image.stem}.txt")
        count += 1
    return count


def choose_balanced(images: list[Path], labels: Path, positives: int, negatives: int, rng: random.Random) -> list[Path]:
    positive_images = [path for path in images if is_positive(path, labels)]
    negative_images = [path for path in images if not is_positive(path, labels)]
    if len(positive_images) < positives or len(negative_images) < negatives:
        raise ValueError(f"Insufficient balanced samples: positives={len(positive_images)}, negatives={len(negative_images)}")
    return rng.sample(positive_images, positives) + rng.sample(negative_images, negatives)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    remote = ROOT / "data" / "combined_sen2gf3_gf" / "processed"
    floodnet = ROOT / "data" / "floodnet_binary_aug_1024" / "processed"
    output = ROOT / "data" / "water_fast_10h" / "processed"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")

    remote_train = sorted((remote / "images" / "train").glob("*"))
    remote_val = sorted((remote / "images" / "val").glob("*"))
    flood_train = sorted((floodnet / "images" / "train").glob("*"))
    flood_val = sorted((floodnet / "images" / "val").glob("*"))

    train_remote = choose_balanced(remote_train, remote / "labels" / "train", positives=2600, negatives=1400, rng=rng)
    val_remote = choose_balanced(remote_val, remote / "labels" / "val", positives=500, negatives=300, rng=rng)

    train_count = add_samples(train_remote, remote / "labels" / "train", "train", "remote", output)
    train_count += add_samples(flood_train, floodnet / "labels" / "train", "train", "floodnet", output)
    val_count = add_samples(val_remote, remote / "labels" / "val", "val", "remote", output)
    val_count += add_samples(flood_val, floodnet / "labels" / "val", "val", "floodnet", output)
    print(f"Built {output}: train={train_count}, val={val_count}")


if __name__ == "__main__":
    main()
