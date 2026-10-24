"""
train_model.py - reproduces the Colab notebook (CKD_datamining_Project.ipynb) and
saves every artifact the Streamlit dashboard needs.

Run once (and again only if you want to retrain):

    python train_model.py                    # downloads UCI CKD dataset (id=336)
    python train_model.py --csv my_ckd.csv   # offline alternative (needs a 'class' column)

What is reproduced from the notebook
    * Dataset: UCI CKD via ucimlrepo (id=336), features + 'class' target
    * Cleaning: strip column names, drop duplicates, strip text values, '?' -> NaN
    * Split: train_test_split(test_size=0.20, random_state=42, stratify=y_encoded)
    * Preprocessing: numeric = median impute + StandardScaler;
                     categorical = most-frequent impute + OneHotEncoder(handle_unknown="ignore")
    * Feature selection: SelectKBest(mutual_info_classif, k=min(15, n_features))
    * Models: Decision Tree (max_depth=5), Random Forest (200 trees), Logistic Regression,
              SVM - all class_weight="balanced"
    * Metrics: accuracy + precision/recall/F1 with CKD as the positive class

Leakage control: the imputers, scaler, encoder and SelectKBest live INSIDE each sklearn
Pipeline and are fitted on X_train only. The test set is only used for evaluation.

Small, documented differences from the notebook (none change the method):
    * numpy is imported (the notebook used np without importing it)
    * SelectKBest uses mutual_info_classif(random_state=42) so the selection is reproducible
    * ColumnTransformer(sparse_threshold=0) guarantees dense output for mutual information
    * every model is trained with AND without feature selection as complete Pipelines, so
      each one can be saved and used by the prediction demo
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from functools import partial

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import ckd_common as cc

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K = 15


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_raw_dataset(csv_path: str | None, target: str, drop: list[str]):
    if csv_path:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        except Exception as exc:  # noqa: BLE001
            fail(f"Could not read '{csv_path}': {exc}")
        if drop:
            df = df.drop(columns=[c for c in drop if c in df.columns])
        if target != cc.TARGET_COL:
            if target not in df.columns:
                fail(f"Target column '{target}' not found in {csv_path}.")
            df = df.rename(columns={target: cc.TARGET_COL})
        return df, f"local CSV: {csv_path}"

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        fail("ucimlrepo is not installed. Run: pip install -r requirements.txt")
    try:
        ckd = fetch_ucirepo(id=336)  # official UCI CKD dataset (same as the notebook)
        df = pd.concat([ckd.data.features, ckd.data.targets], axis=1)
    except Exception as exc:  # noqa: BLE001
        fail("Could not download the UCI CKD dataset (check your internet connection).\n"
             f"Details: {exc}\n"
             "Offline option: python train_model.py --csv path/to/ckd.csv")
    return df, "UCI Machine Learning Repository, CKD dataset (id=336) via ucimlrepo"


# --------------------------------------------------------------------------
# 2. Clean data (same order as the notebook)
# --------------------------------------------------------------------------
def is_text(series: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(series)


def clean_dataset(df: pd.DataFrame):
    log = {"raw_rows": int(len(df)), "raw_columns": int(df.shape[1])}
    df = df.copy()

    # 2.2 clean column names
    df.columns = df.columns.astype(str).str.strip()
    if cc.TARGET_COL not in df.columns:
        fail(f"Expected a '{cc.TARGET_COL}' target column. Found: {df.columns.tolist()}")

    # 2.3 remove duplicate rows
    df = df.drop_duplicates().copy()
    log["duplicates_removed"] = log["raw_rows"] - int(len(df))

    # 3.1 strip spaces/tabs from all text columns
    for col in [c for c in df.columns if is_text(df[c])]:
        df[col] = df[col].astype("object").str.strip()

    # 3.3 placeholder missing values ('?' and empty strings) -> real NaN
    df = df.replace(r"^\s*(\?)?\s*$", np.nan, regex=True)

    # Robustness for CSV input: text columns that are really numbers become numeric.
    # (No-op for the ucimlrepo version of the dataset.)
    for col in [c for c in df.columns if is_text(df[c]) and c != cc.TARGET_COL]:
        converted = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() > 0 and converted.notna().sum() == df[col].notna().sum():
            df[col] = converted

    # Make sure text columns hold np.nan (not None/pd.NA) for the imputers
    for col in [c for c in df.columns if is_text(df[c])]:
        df[col] = df[col].astype("object").where(df[col].notna(), np.nan)

    # Rows without a target label cannot be used
    before = len(df)
    df = df[df[cc.TARGET_COL].notna()].copy()
    log["rows_dropped_missing_target"] = int(before - len(df))
    log["rows_after_cleaning"] = int(len(df))
    return df, log


# --------------------------------------------------------------------------
# 3. Pipeline pieces (identical to the notebook)
# --------------------------------------------------------------------------
def build_preprocessor(numerical_cols, categorical_cols) -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("numerical", numerical_pipeline, numerical_cols),
         ("categorical", categorical_pipeline, categorical_cols)],
        sparse_threshold=0,
    )


def build_models() -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVM": SVC(class_weight="balanced"),
    }


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CKD models and save dashboard artifacts.")
    parser.add_argument("--csv", help="Optional local CSV instead of downloading from UCI.")
    parser.add_argument("--target", default=cc.TARGET_COL, help="Target column name in the CSV (default: class).")
    parser.add_argument("--drop", nargs="*", default=[], help="CSV columns to drop (e.g. an id column).")
    args = parser.parse_args()

    cc.ARTIFACT_DIR.mkdir(exist_ok=True)

    print("[1/6] Loading dataset ...")
    raw, source = load_raw_dataset(args.csv, args.target, args.drop)

    print("[2/6] Cleaning dataset ...")
    df, clean_log = clean_dataset(raw)

    X = df.drop(columns=[cc.TARGET_COL])
    y = df[cc.TARGET_COL]
    numerical_cols = [c for c in X.columns if not is_text(X[c])]
    categorical_cols = [c for c in X.columns if is_text(X[c])]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = [str(c) for c in label_encoder.classes_]
    if len(class_names) != 2 or cc.CKD_LABEL not in class_names:
        fail(f"Expected two classes including '{cc.CKD_LABEL}'. Found: {class_names}")
    ckd_index = class_names.index(cc.CKD_LABEL)
    class_labels = list(range(len(class_names)))

    print("[3/6] Splitting data (stratified 80/20, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)

    # ---- Feature scores: preprocessing fitted on TRAINING data only (notebook step 8.3)
    print("[4/6] Scoring features with mutual information (training data only) ...")
    prep_for_scores = build_preprocessor(numerical_cols, categorical_cols)
    X_train_processed = prep_for_scores.fit_transform(X_train)
    raw_feature_names = prep_for_scores.get_feature_names_out()
    mi_scores = mutual_info_classif(X_train_processed, y_train, random_state=RANDOM_STATE)
    k = min(TOP_K, X_train_processed.shape[1])
    fs_full, fs_top = "All features", f"Top {k} features"

    # ---- Train every model with and without feature selection (notebook step 10)
    print("[5/6] Training 4 models x 2 feature sets and evaluating on the held-out test set ...")
    pipelines = {fs_full: {}, fs_top: {}}
    metric_rows, cm_rows = [], []
    selected_raw_names = None

    for name, model in build_models().items():
        for fs in (fs_full, fs_top):
            steps = [("preprocessor", build_preprocessor(numerical_cols, categorical_cols))]
            if fs == fs_top:
                steps.append(("selector", SelectKBest(
                    score_func=partial(mutual_info_classif, random_state=RANDOM_STATE), k=k)))
            steps.append(("classifier", clone(model)))
            pipe = Pipeline(steps).fit(X_train, y_train)      # fitted on training data only
            pipelines[fs][name] = pipe

            if fs == fs_top and selected_raw_names is None:
                selected_raw_names = pipe.named_steps["preprocessor"].get_feature_names_out()[
                    pipe.named_steps["selector"].get_support()].tolist()

            pred = pipe.predict(X_test)
            metric_rows.append({
                "Model": name,
                "Feature Set": fs,
                "Accuracy": accuracy_score(y_test, pred),
                "CKD Precision": precision_score(y_test, pred, pos_label=ckd_index, zero_division=0),
                "CKD Recall": recall_score(y_test, pred, pos_label=ckd_index, zero_division=0),
                "CKD F1": f1_score(y_test, pred, pos_label=ckd_index, zero_division=0),
            })
            cm = confusion_matrix(y_test, pred, labels=class_labels)
            for i, actual in enumerate(class_names):
                for j, predicted in enumerate(class_names):
                    cm_rows.append({"Model": name, "Feature Set": fs, "Actual": actual,
                                    "Predicted": predicted, "Count": int(cm[i, j])})
            print(f"      {name:<20} | {fs:<16} | acc={metric_rows[-1]['Accuracy']:.3f} "
                  f"| CKD F1={metric_rows[-1]['CKD F1']:.3f}")

    comparison_df = pd.DataFrame(metric_rows)
    confusion_df = pd.DataFrame(cm_rows)

    feature_scores = pd.DataFrame({
        "Feature": raw_feature_names,
        "Label": [cc.pretty_feature_name(n) for n in raw_feature_names],
        "Original Attribute": [cc.original_attribute(n, categorical_cols) for n in raw_feature_names],
        "Mutual Information": mi_scores,
        "Selected": [n in set(selected_raw_names) for n in raw_feature_names],
    }).sort_values("Mutual Information", ascending=False).reset_index(drop=True)
    feature_scores.insert(0, "Rank", feature_scores.index + 1)

    # ---- Pick the model to save (notebook step 11: highest CKD F1, all-features pipelines)
    full_rows = comparison_df[comparison_df["Feature Set"] == fs_full].reset_index(drop=True)
    ranked = full_rows.sort_values(["CKD F1", "Accuracy"], ascending=False, kind="mergesort")
    best_name = str(ranked.iloc[0]["Model"])
    top_f1 = ranked.iloc[0]["CKD F1"]
    tied = [str(m) for m in full_rows.loc[np.isclose(full_rows["CKD F1"], top_f1), "Model"]]

    print("[6/6] Saving artifacts ...")
    joblib.dump(pipelines[fs_full][best_name], cc.path("best_model"))
    joblib.dump(label_encoder, cc.path("label_encoder"))
    joblib.dump(pipelines, cc.path("pipelines"))
    comparison_df.to_csv(cc.path("comparison"), index=False)
    confusion_df.to_csv(cc.path("confusion"), index=False)
    feature_scores.to_csv(cc.path("feature_scores"), index=False)
    df.to_csv(cc.path("cleaned"), index=False)
    test_out = X_test.copy()
    test_out[cc.TARGET_COL] = label_encoder.inverse_transform(y_test)
    test_out.to_csv(cc.path("test_set"), index=False)

    # Schema used by the prediction form (statistics from TRAINING data only)
    schema = {
        "feature_columns": X.columns.tolist(),
        "dtypes": {c: str(X[c].dtype) for c in X.columns},
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "numeric_stats": {c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()),
                              "median": float(X_train[c].median()), "mean": float(X_train[c].mean()),
                              "missing_in_train": int(X_train[c].isna().sum())}
                          for c in numerical_cols},
        "categorical_options": {c: sorted(X_train[c].dropna().unique().tolist()) for c in categorical_cols},
        "categorical_mode": {c: (X_train[c].mode().iloc[0] if X_train[c].notna().any() else None)
                             for c in categorical_cols},
    }
    with open(cc.path("schema"), "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)

    counts = lambda arr: {class_names[i]: int((np.asarray(arr) == i).sum()) for i in class_labels}  # noqa: E731
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "top_k": k,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "cleaning": clean_log,
        "n_features": int(X.shape[1]),
        "n_encoded_features": int(X_train_processed.shape[1]),
        "n_selected_features": int(k),
        "class_names": class_names,
        "ckd_label": cc.CKD_LABEL,
        "ckd_index": ckd_index,
        "class_counts": counts(y_encoded),
        "train_class_counts": counts(y_train),
        "test_class_counts": counts(y_test),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "total_missing_cells": int(df.drop(columns=[cc.TARGET_COL]).isna().sum().sum()),
        "feature_sets": {"full": fs_full, "selected": fs_top},
        "models": list(pipelines[fs_full].keys()),
        "selected_features": [cc.pretty_feature_name(n) for n in selected_raw_names],
        "best_saved": {"model": best_name, "feature_set": fs_full,
                       "rule": "highest CKD F1 on the test set among all-feature pipelines "
                               "(ties broken by accuracy, then model order)"},
        "tied_best_models": tied,
    }
    with open(cc.path("summary"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nDone. Artifacts written to:", cc.ARTIFACT_DIR)
    print(f"Saved ckd_model.pkl = {best_name} (all features)"
          + (f"  [tied on CKD F1: {', '.join(tied)}]" if len(tied) > 1 else ""))
    print("Next: streamlit run app.py")


if __name__ == "__main__":
    main()
