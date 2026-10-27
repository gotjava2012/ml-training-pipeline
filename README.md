# ML Training Pipeline (scikit-learn)

A complete, professional machine-learning workflow with a real library: load →
train/test split → preprocess → train → evaluate → persist → predict. Runs
locally, no GPU.

## What it demonstrates
- Proper **train/test split** (and why you evaluate on held-out data)
- A scikit-learn **Pipeline** (scaler + model) so preprocessing is consistent
- **Cross-validation** and honest metrics: accuracy, precision/recall/F1,
  confusion matrix, feature importances
- Model persistence (save/load) and inference on new data

## Quick start
```bash
pip3 install scikit-learn
python3 src/train.py train                      # train, evaluate, save model
python3 src/train.py predict 6.3 3.3 6.0 2.5    # classify a new sample
```

## Learn it step by step
See **[docs/TUTORIAL.md](docs/TUTORIAL.md)** — covers overfitting, why accuracy
alone isn't enough, reproducibility, and how to scale up (XGBoost, tuning, MLflow).

## Project layout
```
src/train.py        # full training pipeline (heavily commented)
docs/TUTORIAL.md    # step-by-step tutorial
```

## Stack
Python · scikit-learn (RandomForest, StandardScaler, Pipeline)
