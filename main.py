from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute eigenfaces-style PCA components for Birkin bag images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("Data/Birkin"),
        help="Directory containing .jpg/.jpeg/.png images.",
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
        default=12,
        help="Number of principal components to compute.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Image index to reconstruct from the PCA basis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where visualizations are saved.",
    )
    return parser.parse_args()


def load_images(image_dir: Path, size: int) -> tuple[np.ndarray, list[Path]]:
    patterns = ("*.jpg", "*.jpeg", "*.png")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(image_dir.glob(pattern)))

    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    vectors = []
    for path in files:
        img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        vectors.append(arr.reshape(-1))

    return np.vstack(vectors), files


def save_component_grid(components: np.ndarray, size: int, output_path: Path) -> None:
    n_components = components.shape[0]
    cols = min(4, n_components)
    rows = int(np.ceil(n_components / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, ax in enumerate(axes.flat):
        if idx < n_components:
            comp = components[idx].reshape(size, size)
            ax.imshow(comp, cmap="gray")
            ax.set_title(f"PC {idx + 1}")
        ax.axis("off")

    fig.suptitle("EigenBirkins (principal components)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_reconstruction(
    images: np.ndarray,
    pca: PCA,
    sample_index: int,
    size: int,
    output_path: Path,
) -> None:
    safe_idx = max(0, min(sample_index, len(images) - 1))

    x = images[safe_idx : safe_idx + 1]
    x_proj = pca.transform(x)
    x_rec = pca.inverse_transform(x_proj)

    original = x[0].reshape(size, size)
    reconstructed = x_rec[0].reshape(size, size)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(reconstructed, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images, files = load_images(args.image_dir, args.size)

    n_components = min(args.components, images.shape[0], images.shape[1])
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=False)
    pca.fit(images)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    mean_image = pca.mean_.reshape(args.size, args.size)
    mean_path = args.output_dir / "mean_birkin.png"
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_image, cmap="gray")
    plt.axis("off")
    plt.title("Mean Birkin")
    plt.tight_layout()
    plt.savefig(mean_path, dpi=180)
    plt.close()

    components_path = args.output_dir / "eigenbirkins.png"
    save_component_grid(pca.components_, args.size, components_path)

    recon_path = args.output_dir / "reconstruction.png"
    save_reconstruction(images, pca, args.sample_index, args.size, recon_path)

    print(f"Loaded {len(files)} images from {args.image_dir}")
    print(f"Computed {n_components} components")
    print(f"Explained variance (first 5): {explained[:5]}")
    print(f"Cumulative explained variance: {cumulative[-1]:.4f}")
    print(f"Saved: {mean_path}")
    print(f"Saved: {components_path}")
    print(f"Saved: {recon_path}")


if __name__ == "__main__":
    main()
