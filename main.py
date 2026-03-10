from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an EigenBags (PCA) + KNN classifier to predict whether a bag image "
            "is Birkin or not-Birkin."
        )
    )
    parser.add_argument(
        "--birkin-dirs",
        nargs="+",
        type=Path,
        default=[Path("Data/Birkin"), Path("Data/birkins")],
        help="One or more directories containing Birkin images.",
    )
    parser.add_argument(
        "--other-dir",
        type=Path,
        default=Path("Data/other"),
        help="Directory containing non-Birkin images.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Square size used for PCA (images are resized to size x size).",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=48,
        help="Number of principal components (eigenbags).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors for KNN.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples used for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Test sample index to visualize reconstruction in eigenbag space.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where visualizations and reports are saved.",
    )
    return parser.parse_args()


def list_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(sorted(image_dir.glob(pattern)))
    return files


def load_image_vector(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def build_dataset(
    birkin_dirs: list[Path],
    other_dir: Path,
    size: int,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    birkin_files: list[Path] = []
    for directory in birkin_dirs:
        birkin_files.extend(list_image_files(directory))

    other_files = list_image_files(other_dir)

    if not birkin_files:
        raise FileNotFoundError("No Birkin images found in provided --birkin-dirs")
    if not other_files:
        raise FileNotFoundError(f"No non-Birkin images found in {other_dir}")

    files = birkin_files + other_files
    y = np.array([1] * len(birkin_files) + [0] * len(other_files), dtype=np.int64)

    x = np.vstack([load_image_vector(path, size) for path in files])
    return x, y, files


def save_component_grid(components: np.ndarray, size: int, output_path: Path) -> None:
    n_components = components.shape[0]
    cols = min(6, n_components)
    rows = int(np.ceil(n_components / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, ax in enumerate(axes.flat):
        if idx < n_components:
            comp = components[idx].reshape(size, size)
            ax.imshow(comp, cmap="gray")
            ax.set_title(f"EigenBag {idx + 1}")
        ax.axis("off")

    fig.suptitle("Top EigenBags (PCA components)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_reconstruction(
    x_original: np.ndarray,
    x_projected: np.ndarray,
    pca: PCA,
    sample_index: int,
    size: int,
    output_path: Path,
) -> None:
    safe_idx = max(0, min(sample_index, len(x_original) - 1))

    original = x_original[safe_idx].reshape(size, size)
    reconstructed = pca.inverse_transform(x_projected[safe_idx : safe_idx + 1])[0].reshape(size, size)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Test Image")
    axes[0].axis("off")

    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].set_title("Reconstruction in EigenBag Space")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y, files = build_dataset(args.birkin_dirs, args.other_dir, args.size)

    x_train, x_test, y_train, y_test, files_train, files_test = train_test_split(
        x,
        y,
        files,
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

    accuracy = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, target_names=["not_birkin", "birkin"], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    mean_path = args.output_dir / "mean_bag.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(pca.mean_.reshape(args.size, args.size), cmap="gray")
    plt.axis("off")
    plt.title("Mean Bag")
    plt.tight_layout()
    plt.savefig(mean_path, dpi=180)
    plt.close()

    components_path = args.output_dir / "eigenbags.png"
    save_component_grid(pca.components_, args.size, components_path)

    recon_path = args.output_dir / "test_reconstruction.png"
    save_reconstruction(x_test, x_test_pca, pca, args.sample_index, args.size, recon_path)

    report_path = args.output_dir / "classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("EigenBags + KNN classification\n")
        f.write(f"total_samples={len(x)}\n")
        f.write(f"birkin_samples={int(y.sum())}\n")
        f.write(f"not_birkin_samples={len(y) - int(y.sum())}\n")
        f.write(f"train_samples={len(x_train)}\n")
        f.write(f"test_samples={len(x_test)}\n")
        f.write(f"image_size={args.size}x{args.size}\n")
        f.write(f"n_components={n_components}\n")
        f.write(f"k={args.k}\n")
        f.write(f"accuracy={accuracy:.4f}\n\n")
        f.write("Confusion matrix [[TN, FP], [FN, TP]]:\n")
        f.write(f"{cm}\n\n")
        f.write(report)

    print("=== Dataset composition ===")
    print(f"Total images: {len(x)}")
    print(f"Birkin (from {', '.join(str(d) for d in args.birkin_dirs)}): {int(y.sum())}")
    print(f"Not Birkin (from {args.other_dir}): {len(y) - int(y.sum())}")
    print("=== Model results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(cm)
    print(report)
    print("=== Outputs ===")
    print(f"Saved: {mean_path}")
    print(f"Saved: {components_path}")
    print(f"Saved: {recon_path}")
    print(f"Saved: {report_path}")
    print(f"Example held-out file: {files_test[min(args.sample_index, len(files_test) - 1)]}")


if __name__ == "__main__":
    main()
