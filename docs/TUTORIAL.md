# Tutorial: A Real ML Training Pipeline (Step by Step)

Where the [neural-net-from-scratch](../../neural-net-from-scratch) project shows
the *mechanics* of training by hand, this one shows the **professional workflow**
every ML practitioner uses — with a real library (**scikit-learn**). By the end
you'll understand train/test splits, pipelines, evaluation metrics, overfitting,
and model persistence.

Everything here **runs locally and is verified** — no GPU needed.

---

## Prerequisites
```bash
pip3 install scikit-learn
```

## Run it
```bash
python3 src/train.py train                      # train + evaluate + save model
python3 src/train.py predict 6.3 3.3 6.0 2.5    # classify a new flower
```

---

## The 7 steps (map each to `src/train.py`)

### 1. Load data
`load_iris()` gives features `X` (flower measurements) and labels `y` (species).

### 2. Train/test split — the most important concept here
`train_test_split(..., test_size=0.25, stratify=y)` holds out 25% of the data
that the model will **never see during training**. We measure success on that
held-out set, because scoring on data the model trained on tells you nothing
about how it handles *new* data. `stratify=y` keeps class proportions balanced
across the split.

### 3-4. Pipeline: preprocess + model together
`Pipeline([...])` bundles the **StandardScaler** (feature scaling) with the
**RandomForestClassifier**. Bundling matters: it guarantees the exact same
preprocessing is applied at training *and* prediction time — forgetting this is
a classic real-world bug.

### 5. Evaluate — honest metrics
- **Cross-validation** (`cross_val_score`, 5-fold) averages over several splits
  for a robust estimate.
- **Held-out test accuracy** is the headline number.
- **Precision / recall / F1 per class** (classification report) shows *where*
  the model is strong or weak, not just overall accuracy.
- **Confusion matrix** shows exactly which classes get confused for which.
- **Feature importances** show which inputs the model relied on (here: petal
  length & width dominate — interpretability).

### 6. Persist
`pickle.dump(...)` saves the trained pipeline to `model.pkl` so you can reuse it
without retraining.

### 7. Predict
`predict()` loads the saved model and classifies a new flower, returning the
class and probabilities.

---

## Key lessons this teaches

- **Overfitting:** a model can score ~100% on training data yet fail on new data.
  The train/test split is how you catch it. (Try removing the split and scoring
  on training data — the number looks great but means nothing.)
- **Accuracy isn't enough:** precision/recall/F1 and the confusion matrix tell
  you *how* a model is right or wrong, which matters when some errors cost more
  than others.
- **Reproducibility:** `random_state=42` everywhere makes runs repeatable.

---

## Exercises
1. Swap `RandomForestClassifier` for `LogisticRegression` or `SVC`. Compare test
   accuracy — different models suit different data.
2. Change `test_size` to 0.5. Does less training data hurt accuracy?
3. Remove `StandardScaler` from the pipeline. Which models care about scaling?
4. Try a harder bundled dataset: `load_wine()` or `load_breast_cancer()`.

---

## How this scales up
- Swap the model for a **gradient-boosted** one (XGBoost/LightGBM) for tabular
  data at scale.
- For deep learning on images/text, move to **PyTorch** (see the companion
  `pytorch-image-classifier` project — GPU/Colab).
- Add **hyperparameter tuning** (`GridSearchCV`) and **experiment tracking**
  (MLflow) for production ML.
