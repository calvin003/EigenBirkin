from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full Birkin vs not-Birkin pipeline: mix data, split train/test, PCA eigenbags, KNN classify."
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
    parser.add_argument("--size", type=int, default=64, help="Image resize size (size x size).")
    parser.add_argument("--components", type=int, default=24, help="Number of PCA components (eigenbags).")
    parser.add_argument("--k", type=int, default=3, help="Number of KNN neighbors.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/pipeline_report.txt"),
        help="Where to save the text report.",
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


def build_dataset(birkin_dirs: list[Path], other_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    birkin_files: list[Path] = []
    for birkin_dir in birkin_dirs:
        birkin_files.extend(list_image_files(birkin_dir))

    other_files = list_image_files(other_dir)

    if not birkin_files:
        raise FileNotFoundError("No Birkin images found in --birkin-dirs")
    if not other_files:
        raise FileNotFoundError(f"No non-Birkin images found in {other_dir}")

    files = birkin_files + other_files
    labels = np.array([1] * len(birkin_files) + [0] * len(other_files), dtype=np.int64)
    features = np.vstack([load_image_vector(path, size) for path in files])
    return features, labels


def main() -> None:
    args = parse_args()

    x, y = build_dataset(args.birkin_dirs, args.other_dir, args.size)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    n_components = min(args.components, x_train.shape[0], x_train.shape[1])
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=args.seed)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    knn = KNeighborsClassifier(n_neighbors=args.k, weights="distance")
    knn.fit(x_train_pca, y_train)
    y_pred = knn.predict(x_test_pca)

    accuracy = float((y_pred == y_test).mean())
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["not_birkin", "birkin"], digits=4)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as file:
        file.write("Birkin classification pipeline (PCA eigenbags + KNN)\n")
        file.write(f"total_samples={len(x)}\n")
        file.write(f"birkin_samples={int(y.sum())}\n")
        file.write(f"not_birkin_samples={len(y) - int(y.sum())}\n")
        file.write(f"train_samples={len(x_train)}\n")
        file.write(f"test_samples={len(x_test)}\n")
        file.write(f"image_size={args.size}x{args.size}\n")
        file.write(f"n_components={n_components}\n")
        file.write(f"k={args.k}\n")
        file.write(f"accuracy={accuracy:.4f}\n\n")
        file.write("Confusion matrix [[TN, FP], [FN, TP]]:\n")
        file.write(f"{cm}\n\n")
        file.write(report)

    print(f"Accuracy: {accuracy:.4f}")
    print(cm)
    print(report)
    print(f"Saved report: {args.report_path}")


if __name__ == "__main__":
    main()
