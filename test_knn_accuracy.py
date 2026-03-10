from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test KNN accuracy using PCA eigenbag features from a train dataset on a held-out test dataset."
    )
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path("outputs/train_dataset.npz"),
        help="Train NPZ produced by create_test_dataset.py",
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=Path("outputs/test_dataset.npz"),
        help="Test NPZ produced by create_test_dataset.py",
    )
    parser.add_argument("--components", type=int, default=48, help="Number of PCA components.")
    parser.add_argument("--k", type=int, default=5, help="Number of KNN neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by PCA.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/knn_test_accuracy_report.txt"),
        help="Where to save evaluation report.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def main() -> None:
    args = parse_args()
    x_train, y_train = load_npz(args.train_dataset)
    x_test, y_test = load_npz(args.test_dataset)

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
        file.write("KNN accuracy test using PCA eigenbag features\n")
        file.write(f"train_dataset={args.train_dataset}\n")
        file.write(f"test_dataset={args.test_dataset}\n")
        file.write(f"train_samples={len(x_train)}\n")
        file.write(f"test_samples={len(x_test)}\n")
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
