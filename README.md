# EigenBirkin

This repo now separates the workflow into scripts for eigenbag generation, dataset mixing, and KNN classification/testing.

1. **`main.py`** (also runnable as `eigenbag_generator.py`) for eigenbag generation/visualization only.
2. **`data_mixer.py`** to mix `Data/Birkin`, `Data/birkins`, and `Data/other` into one labeled dataset.
3. **`knn_classifier.py`** for quick train/evaluate KNN classification using PCA eigenbag features from one dataset.
4. **`create_test_dataset.py`** to create explicit train/test dataset files.
5. **`test_knn_accuracy.py`** to test KNN accuracy using PCA eigenbags learned from train split and evaluated on test split.

## Install

```bash
pip install -r requirements.txt
```

## 1) Eigenbag generator (original flow)

```bash
python main.py --image-dir Data/Birkin --size 128 --components 12
# or
python eigenbag_generator.py --image-dir Data/Birkin --size 128 --components 12
```

Outputs:

- `outputs/mean_birkin.png`
- `outputs/eigenbirkins.png`
- `outputs/reconstruction.png`

## 2) Mix dataset (both Birkin folders + other)

```bash
python data_mixer.py \
  --birkin-dirs Data/Birkin Data/birkins \
  --other-dir Data/other \
  --size 128 \
  --output outputs/mixed_dataset.npz
```

## 3) (Optional) One-command train/evaluate KNN on mixed dataset

```bash
python knn_classifier.py \
  --dataset outputs/mixed_dataset.npz \
  --components 48 \
  --k 5
```

Output:

- `outputs/classification_report.txt`

## 4) Create explicit train/test datasets

```bash
python create_test_dataset.py \
  --dataset outputs/mixed_dataset.npz \
  --test-size 0.2 \
  --train-output outputs/train_dataset.npz \
  --test-output outputs/test_dataset.npz
```

## 5) Test KNN accuracy using PCA eigenbags (train -> test)

```bash
python test_knn_accuracy.py \
  --train-dataset outputs/train_dataset.npz \
  --test-dataset outputs/test_dataset.npz \
  --components 48 \
  --k 5 \
  --report-path outputs/knn_test_accuracy_report.txt
```

Output:

- `outputs/knn_test_accuracy_report.txt`
