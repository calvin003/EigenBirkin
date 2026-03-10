# EigenBirkin

This project builds an **EigenBags + KNN** classifier that predicts whether an image is a **Birkin** or **not Birkin**.

## What it does

1. Loads and mixes datasets from:
   - `Data/Birkin`
   - `Data/birkins`
   - `Data/other` (negative class)
2. Resizes each image and converts to grayscale.
3. Learns PCA components ("eigenbags") on the training split.
4. Projects train/test images into eigenbag space.
5. Trains a KNN classifier and evaluates Birkin vs not-Birkin.
6. Saves visual outputs and a classification report.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py \
  --birkin-dirs Data/Birkin Data/birkins \
  --other-dir Data/other \
  --size 128 \
  --components 48 \
  --k 5
```

## Output files

Saved in `outputs/`:

- `mean_bag.png` - mean image learned by PCA.
- `eigenbags.png` - top PCA components visualized as eigenbags.
- `test_reconstruction.png` - reconstruction of a held-out test sample.
- `classification_report.txt` - dataset composition + confusion matrix + precision/recall/F1.

## Key parameters

- `--components`: number of eigenbags used as features.
- `--k`: number of neighbors for KNN.
- `--test-size`: fraction reserved for evaluation.
- `--seed`: random seed for reproducible split.
