"""
Split class-folder images into train/val/test directories with an 80/10/10 ratio.

Expected input structure:
input_dir/
  class_a/
    img1.jpg
    img2.jpg
  class_b/
    ...

Output structure:
output_root/dataset_name/
  train/class_a/
  val/class_a/
  test/class_a/
  train/class_b/
  ...
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def compute_split_counts(total: int) -> Tuple[int, int, int]:
    """Return counts for train, val, test using an 80/10/10 split."""
    train_count = int(total * 0.8)
    val_count = int(total * 0.1)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def split_and_move(input_dir: Path, output_root: Path, dataset_name: str, seed: int) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    random.seed(seed)
    output_base = output_root / dataset_name
    output_base.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found in input directory: {input_dir}")

    moved_total = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [p for p in class_dir.iterdir() if is_image_file(p)]

        if not images:
            print(f"Skipping class '{class_name}' (no images found).")
            continue

        random.shuffle(images)
        train_count, val_count, test_count = compute_split_counts(len(images))

        split_files = {
            "train": images[:train_count],
            "val": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:],
        }

        for split in SPLITS:
            target_dir = output_base / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for src in split_files[split]:
                destination = target_dir / src.name

                # Prevent accidental overwrite if duplicate filenames already exist.
                if destination.exists():
                    destination = target_dir / f"{src.stem}_{src.stat().st_mtime_ns}{src.suffix}"

                shutil.move(str(src), str(destination))
                moved_total += 1

        print(
            f"{class_name}: total={len(images)}, "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    print(f"\nDone. Moved {moved_total} image(s) into: {output_base}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split class folders into train/val/test and move images."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory that contains class folders.",
    )
    parser.add_argument(
        "--output-root",
        default="for_training",
        help="Base output directory (default: for_training).",
    )
    parser.add_argument(
        "--dataset-name",
        default="default",
        help="Subfolder name under output-root (default: default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before split (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_and_move(
        input_dir=Path(args.input),
        output_root=Path(args.output_root),
        dataset_name=args.dataset_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
