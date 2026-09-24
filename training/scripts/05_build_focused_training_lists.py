#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build YOLO image-list datasets for Sen2GF3-focused fine-tuning.")
    p.add_argument("--processed_dir", default="data/combined_sen2gf3_gf/processed")
    p.add_argument("--output_dir", default="data/focused_sen2gf3_replay")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sen_positive_repeat", type=int, default=3)
    p.add_argument("--gf_replay_ratio", type=float, default=0.25)
    p.add_argument("--stabilize_sen_positive_repeat", type=int, default=2)
    p.add_argument("--stabilize_gf_ratio", type=float, default=0.50)
    return p.parse_args()


def image_paths(processed_dir: Path, split: str, prefix: str) -> list[Path]:
    image_dir = processed_dir / "images" / split
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(image_dir.glob(f"{prefix}*{ext[1:]}"))
    return sorted(paths)


def is_positive(processed_dir: Path, split: str, image_path: Path) -> bool:
    label_path = processed_dir / "labels" / split / f"{image_path.stem}.txt"
    return label_path.exists() and label_path.stat().st_size > 0


def write_list(path: Path, paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p.resolve()) for p in paths) + "\n", encoding="ascii")


def write_yaml(path: Path, processed_dir: Path, train: str, val: str, test: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"path: {processed_dir.resolve()}",
                f"train: {train}",
                f"val: {val}",
                f"test: {test}",
                "names:",
                "  0: water",
                "",
            ]
        ),
        encoding="ascii",
    )


def sample_ratio(paths: list[Path], ratio: float, rng: random.Random) -> list[Path]:
    if ratio >= 1:
        return list(paths)
    n = max(1, round(len(paths) * ratio))
    return sorted(rng.sample(paths, n))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    processed_dir = Path(args.processed_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sen_train = image_paths(processed_dir, "train", "sen2gf3_")
    gf_train = image_paths(processed_dir, "train", "gf_")
    sen_val = image_paths(processed_dir, "val", "sen2gf3_")
    gf_val = image_paths(processed_dir, "val", "gf_")
    sen_test = image_paths(processed_dir, "test", "sen2gf3_")
    gf_test = image_paths(processed_dir, "test", "gf_")

    sen_train_pos = [p for p in sen_train if is_positive(processed_dir, "train", p)]
    gf_replay = sample_ratio(gf_train, args.gf_replay_ratio, rng)
    stabilize_gf = sample_ratio(gf_train, args.stabilize_gf_ratio, rng)

    focus_train = sen_train + sen_train_pos * max(0, args.sen_positive_repeat - 1) + gf_replay
    rng.shuffle(focus_train)

    stabilize_train = (
        sen_train
        + sen_train_pos * max(0, args.stabilize_sen_positive_repeat - 1)
        + stabilize_gf
    )
    rng.shuffle(stabilize_train)

    write_list(out / "train_sen2gf3_focus_replay.txt", focus_train)
    write_list(out / "train_sen2gf3_gf_stabilize.txt", stabilize_train)
    write_list(out / "val_sen2gf3.txt", sen_val)
    write_list(out / "val_gf_floodnet.txt", gf_val)
    write_list(out / "val_combined.txt", sen_val + gf_val)
    write_list(out / "test_sen2gf3.txt", sen_test)
    write_list(out / "test_gf_floodnet.txt", gf_test)
    write_list(out / "test_combined.txt", sen_test + gf_test)

    write_yaml(
        out / "waterlogging_sen2gf3_focus_replay.yaml",
        processed_dir,
        str((out / "train_sen2gf3_focus_replay.txt").resolve()),
        str((out / "val_sen2gf3.txt").resolve()),
        str((out / "test_sen2gf3.txt").resolve()),
    )
    write_yaml(
        out / "waterlogging_sen2gf3_gf_stabilize.yaml",
        processed_dir,
        str((out / "train_sen2gf3_gf_stabilize.txt").resolve()),
        str((out / "val_combined.txt").resolve()),
        str((out / "test_combined.txt").resolve()),
    )
    write_yaml(
        out / "waterlogging_eval_sen2gf3.yaml",
        processed_dir,
        str((out / "train_sen2gf3_focus_replay.txt").resolve()),
        str((out / "val_sen2gf3.txt").resolve()),
        str((out / "test_sen2gf3.txt").resolve()),
    )
    write_yaml(
        out / "waterlogging_eval_gf_floodnet.yaml",
        processed_dir,
        str((out / "train_sen2gf3_focus_replay.txt").resolve()),
        str((out / "val_gf_floodnet.txt").resolve()),
        str((out / "test_gf_floodnet.txt").resolve()),
    )

    print("Sen2GF3 train:", len(sen_train))
    print("Sen2GF3 train positives:", len(sen_train_pos))
    print("GF-FloodNet train:", len(gf_train))
    print("GF replay selected:", len(gf_replay))
    print("Phase1 focus train list:", len(focus_train))
    print("Phase2 stabilize train list:", len(stabilize_train))
    print("Sen2GF3 val/test:", len(sen_val), len(sen_test))
    print("GF-FloodNet val/test:", len(gf_val), len(gf_test))
    print("Output:", out.resolve())


if __name__ == "__main__":
    main()
