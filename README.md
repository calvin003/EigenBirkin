# EigenBirkin

Simple, single-command pipeline for **Birkin vs not-Birkin** classification using **PCA eigenbags + KNN**.

## Install

```bash
pip install -r requirements.txt
```

## Run (all-in-one)

```bash
python run_pipeline.py
```

This single script does everything:
- mixes data from `Data/Birkin` + `Data/birkins` (Birkin) and `Data/other` (not-Birkin)
- splits into train/test
- learns PCA eigenbag features on train set
- trains KNN classifier
- evaluates accuracy on test set
- writes report to `outputs/pipeline_report.txt`

## Optional parameters

```bash
python run_pipeline.py \
  --birkin-dirs Data/Birkin Data/birkins \
  --other-dir Data/other \
  --size 64 \
  --components 24 \
  --k 3 \
  --test-size 0.2 \
  --seed 42 \
  --report-path outputs/pipeline_report.txt
```
