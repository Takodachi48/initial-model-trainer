import argparse
from typing import Dict, List, Tuple

import torch


def load_label_metadata(checkpoint_path: str, map_location: str = "cpu") -> Tuple[Dict[str, int], List[str], Dict[int, str]]:
    """
    Load label metadata from a checkpoint file.

    Returns:
        Tuple of (label_mapping, class_names, idx_to_class)
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    label_mapping = checkpoint.get("label_mapping", {}) or {}
    class_names = checkpoint.get("class_names", []) or []
    idx_to_class_raw = checkpoint.get("idx_to_class", {}) or {}

    # Normalize types for consistency across saved checkpoints.
    normalized_label_mapping = {str(k): int(v) for k, v in label_mapping.items()}
    normalized_idx_to_class = {}
    for k, v in idx_to_class_raw.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        normalized_idx_to_class[idx] = str(v)

    if not normalized_idx_to_class and normalized_label_mapping:
        normalized_idx_to_class = {idx: name for name, idx in normalized_label_mapping.items()}

    if not class_names and normalized_label_mapping:
        class_names = [name for name, _ in sorted(normalized_label_mapping.items(), key=lambda kv: kv[1])]

    return normalized_label_mapping, [str(c) for c in class_names], normalized_idx_to_class


def print_label_metadata(checkpoint_path: str, map_location: str = "cpu") -> None:
    """Print label metadata in a compact human-readable format."""
    label_mapping, class_names, idx_to_class = load_label_metadata(checkpoint_path, map_location=map_location)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Classes: {len(class_names) if class_names else len(label_mapping)}")

    if class_names:
        print("class_names:")
        for idx, name in enumerate(class_names):
            print(f"  {idx}: {name}")

    if label_mapping:
        print("label_mapping:")
        for name, idx in sorted(label_mapping.items(), key=lambda kv: kv[1]):
            print(f"  {name} -> {idx}")
    else:
        print("label_mapping: <missing>")

    if idx_to_class:
        print("idx_to_class:")
        for idx, name in sorted(idx_to_class.items(), key=lambda kv: kv[0]):
            print(f"  {idx} -> {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect label mapping metadata in a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--map_location", type=str, default="cpu", help="torch.load map_location (default: cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    print_label_metadata(args.checkpoint, map_location=args.map_location)


if __name__ == "__main__":
    main()
