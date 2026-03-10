from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix Birkin and non-Birkin folders into a single labeled dataset file."
    )
    parser.add_argument(
        "--birkin-dirs",
        nargs="+",
        type=Path,
        default=[Path("Data/Birkin"), Path("Data/birkins")],
        help="Directories containing Birkin images.",
    )
    parser.add_argument(
        "--other-dir",
        type=Path,
        default=Path("Data/other"),
        help="Directory containing non-Birkin images.",
    )
    parser.add_argument("--size", type=int, default=128, help="Resize images to size x size.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mixed_dataset.npz"),
        help="Output NPZ dataset path.",
    )
    return parser.parse_args()


def list_image_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(sorted(folder.glob(pattern)))
    return files


def load_image_vector(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return (np.asarray(image, dtype=np.float32) / 255.0).reshape(-1)


def main() -> None:
    args = parse_args()

    birkin_files: list[Path] = []
    for birkin_dir in args.birkin_dirs:
        birkin_files.extend(list_image_files(birkin_dir))

    other_files = list_image_files(args.other_dir)

    if not birkin_files:
        raise FileNotFoundError("No Birkin images found in --birkin-dirs")
    if not other_files:
        raise FileNotFoundError(f"No non-Birkin images found in {args.other_dir}")

    files = birkin_files + other_files
    labels = np.array([1] * len(birkin_files) + [0] * len(other_files), dtype=np.int64)
    features = np.vstack([load_image_vector(path, args.size) for path in files])
    paths = np.array([str(path) for path in files], dtype=object)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=features,
        y=labels,
        paths=paths,
        size=np.array([args.size], dtype=np.int64),
    )

    print(f"Saved dataset: {args.output}")
    print(f"Total samples: {len(features)}")
    print(f"Birkin samples: {int(labels.sum())}")
    print(f"Not-Birkin samples: {len(labels) - int(labels.sum())}")


if __name__ == "__main__":
    main()
