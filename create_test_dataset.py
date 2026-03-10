from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create explicit train/test NPZ datasets from a mixed dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/mixed_dataset.npz"),
        help="Input dataset created by data_mixer.py",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for held-out test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("outputs/train_dataset.npz"),
        help="Path to save train split.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("outputs/test_dataset.npz"),
        help="Path to save test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}. Run data_mixer.py first.")

    data = np.load(args.dataset, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    paths = data["paths"]
    size = data["size"]

    x_train, x_test, y_train, y_test, p_train, p_test = train_test_split(
        x,
        y,
        paths,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(args.train_output, X=x_train, y=y_train, paths=p_train, size=size)
    np.savez_compressed(args.test_output, X=x_test, y=y_test, paths=p_test, size=size)

    print(f"Saved train split: {args.train_output} ({len(x_train)} samples)")
    print(f"Saved test split: {args.test_output} ({len(x_test)} samples)")


if __name__ == "__main__":
    main()
