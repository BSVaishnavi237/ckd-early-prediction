# CKD Early Prediction & Feature Selection – Streamlit Dashboard

B.Tech Data Mining project dashboard built on the Colab notebook `CKD_datamining_Project.ipynb`
(UCI Chronic Kidney Disease dataset, id=336). **Educational prototype – not a medical diagnostic tool.**

## Run it
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py        # once: downloads UCI data, trains, writes artifacts/
streamlit run app.py         # opens http://localhost:8501
```
Offline? `python train_model.py --csv your_ckd.csv` (needs a `class` column; use `--target` / `--drop` if yours differs).

## Files
| File | Purpose |
|---|---|
| `train_model.py` | Reproduces the notebook, saves pipelines + real metrics/scores to `artifacts/` |
| `app.py` | Dashboard: Overview, Feature Analysis, Model Comparison, Prediction Demo |
| `ckd_common.py` | Shared constants, file names, form metadata |
| `artifacts/` | Created by training: `ckd_model.pkl`, `ckd_label_encoder.pkl`, `ckd_pipelines.pkl`, `model_comparison.csv`, `feature_scores.csv`, `confusion_matrices.csv`, `cleaned_dataset.csv`, `test_set.csv`, `schema.json`, `run_summary.json` |

## What comes from the notebook
Cleaning (strip names/text, drop duplicates, `?` → NaN) · stratified 80/20 split, `random_state=42` ·
median+StandardScaler / most-frequent+OneHotEncoder · `SelectKBest(mutual_info_classif, k=15)` ·
Decision Tree, Random Forest, Logistic Regression, SVM (`class_weight="balanced"`) ·
accuracy, CKD precision/recall/F1 (CKD = positive class), confusion matrices ·
`ckd_model.pkl` = all-feature model with the highest CKD F1.

## Integrity notes
- All dashboard numbers are read from files written by `train_model.py`; nothing is hardcoded.
- No leakage: imputers, scaler, encoder and SelectKBest sit inside each Pipeline and are fitted on the training split only.
- The app never retrains; it loads artifacts (cached). Retrain only by rerunning `train_model.py`.
- Small deviations from the notebook: `numpy` imported (notebook omitted it); mutual information seeded (`random_state=42`) inside SelectKBest for reproducibility; `sparse_threshold=0` so mutual information always sees dense data; each model is saved as a full Pipeline both with and without selection.
- Model "best" is judged on one small test split. Ties are shown as ties; do not read them as proof of superiority.
- If scikit-learn versions differ between training and running, the app warns; rerun `train_model.py`.
