"""
ML Training Pipeline (scikit-learn) — a complete, real training workflow.

Unlike the from-scratch NumPy project (which shows the *mechanics* of training),
this project shows the *professional workflow* every ML practitioner follows,
using a real industry library:

    1. LOAD        - get a dataset
    2. SPLIT       - train/test split so we measure TRUE generalization
    3. PREPROCESS  - scale features (a pipeline step)
    4. TRAIN       - fit a model on the training set
    5. EVALUATE    - score on the HELD-OUT test set (accuracy, precision, recall)
    6. PERSIST     - save the trained model to disk
    7. PREDICT     - load the model and make predictions on new data

The key lesson: you evaluate on data the model NEVER saw during training. A model
that scores 100% on its training data but poorly on the test set is overfitting.

Dataset: the classic Iris flower dataset (bundled with scikit-learn, no download).
"""

import pathlib
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODEL_PATH = pathlib.Path(__file__).parent.parent / "model.pkl"


def load_data():
    """Step 1 — load the dataset (features X, labels y)."""
    data = load_iris()
    return data.data, data.target, data.target_names, data.feature_names


def build_pipeline():
    """Steps 3-4 — a Pipeline bundles preprocessing + model so the SAME steps
    apply consistently at train and predict time (prevents a common bug where
    training and inference preprocess differently)."""
    return Pipeline([
        ("scaler", StandardScaler()),                 # standardize features
        ("clf", RandomForestClassifier(
            n_estimators=100, random_state=42)),      # the model
    ])


def train_and_evaluate():
    X, y, class_names, feature_names = load_data()

    # Step 2 — hold out 25% of data for an honest test of generalization.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    pipe = build_pipeline()

    # Cross-validation on the training set: a more robust estimate than a single
    # split, because it averages over several folds.
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"5-fold CV accuracy (train): {cv_scores.mean()*100:.1f}% "
          f"(+/- {cv_scores.std()*100:.1f}%)")

    # Step 4 — fit on the training set.
    pipe.fit(X_train, y_train)

    # Step 5 — evaluate on the HELD-OUT test set (the honest number).
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nHeld-out TEST accuracy: {acc*100:.1f}%\n")
    print("Classification report (precision/recall/F1 per class):")
    print(classification_report(y_test, preds, target_names=list(class_names)))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, preds))

    # Which features mattered most? (interpretability)
    importances = pipe.named_steps["clf"].feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name:22s} {imp:.3f}")

    # Step 6 — persist the trained pipeline.
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipe, "class_names": list(class_names),
                     "feature_names": list(feature_names)}, f)
    print(f"\nSaved trained model -> {MODEL_PATH}")
    return acc


def predict(features):
    """Step 7 — load the saved model and classify a new flower.
    `features` = [sepal_length, sepal_width, petal_length, petal_width] in cm."""
    if not MODEL_PATH.exists():
        raise SystemExit("No model found. Run:  python3 src/train.py train")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    pipe = bundle["pipeline"]
    pred = pipe.predict([features])[0]
    proba = pipe.predict_proba([features])[0]
    name = bundle["class_names"][pred]
    print(f"Input features: {features}")
    print(f"Predicted class: {name}")
    print("Class probabilities:")
    for cname, p in zip(bundle["class_names"], proba):
        print(f"  {cname:12s} {p*100:5.1f}%")
    return name


def main():
    import sys
    args = sys.argv[1:]
    if not args or args[0] == "train":
        train_and_evaluate()
    elif args[0] == "predict":
        # example: python3 src/train.py predict 5.1 3.5 1.4 0.2
        feats = [float(x) for x in args[1:5]] if len(args) >= 5 else [5.1, 3.5, 1.4, 0.2]
        predict(feats)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
