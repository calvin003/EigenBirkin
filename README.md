# EigenBirkin

This repo now separates the workflow into 3 scripts:

1. **`main.py`** (also runnable as `eigenbag_generator.py`) for eigenbag generation/visualization only.
2. **`data_mixer.py`** for mixing `Data/Birkin`, `Data/birkins`, and `Data/other` into one labeled dataset.
3. **`knn_classifier.py`** for Birkin vs not-Birkin classification using PCA eigenbags + KNN.

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

## 3) Train/evaluate KNN on eigenbag features

```bash
python knn_classifier.py \
  --dataset outputs/mixed_dataset.npz \
  --components 48 \
  --k 5
```

Output:

- `outputs/classification_report.txt`
