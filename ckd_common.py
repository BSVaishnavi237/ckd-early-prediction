"""Shared constants and helpers for the CKD dashboard (used by train_model.py and app.py)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

TARGET_COL = "class"      # target column name in the UCI CKD dataset (id=336)
CKD_LABEL = "ckd"         # label of the positive (disease) class

FILES = {
    "best_model": "ckd_model.pkl",              # best full-feature pipeline (same name as the notebook)
    "label_encoder": "ckd_label_encoder.pkl",   # same name as the notebook
    "pipelines": "ckd_pipelines.pkl",           # every trained pipeline {feature_set: {model: pipeline}}
    "comparison": "model_comparison.csv",
    "feature_scores": "feature_scores.csv",
    "confusion": "confusion_matrices.csv",
    "cleaned": "cleaned_dataset.csv",
    "test_set": "test_set.csv",
    "schema": "schema.json",
    "summary": "run_summary.json",
}
REQUIRED_FILES = list(FILES.values())


def path(key: str) -> Path:
    return ARTIFACT_DIR / FILES[key]


# --------------------------------------------------------------------------
# Feature-name helpers
# --------------------------------------------------------------------------
def pretty_feature_name(raw: str) -> str:
    """'numerical__age' -> 'age', 'categorical__rbc_normal' -> 'rbc_normal'."""
    for prefix in ("numerical__", "categorical__"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def original_attribute(raw: str, categorical_cols) -> str:
    """Map an encoded feature name back to the original dataset column."""
    name = pretty_feature_name(raw)
    if raw.startswith("categorical__"):
        matches = [c for c in categorical_cols if name == c or name.startswith(c + "_")]
        if matches:
            return max(matches, key=len)
    return name


# --------------------------------------------------------------------------
# Form metadata for the prediction demo.
# 'min'/'max' are INPUT-VALIDATION limits (physiologically plausible ranges),
# not model results. Descriptions follow the UCI CKD dataset documentation.
# --------------------------------------------------------------------------
GROUP_ORDER = ["Demographics & vitals", "Urine tests", "Blood tests", "History & symptoms"]

FIELD_META = {
    "age":  dict(label="Age", unit="years", group="Demographics & vitals", min=1.0, max=120.0, step=1.0, fmt="%.0f", desc="Patient age in years."),
    "bp":   dict(label="Blood pressure", unit="mm Hg", group="Demographics & vitals", min=30.0, max=250.0, step=10.0, fmt="%.0f", desc="Blood pressure (diastolic) in mm Hg."),
    "sg":   dict(label="Specific gravity", unit="", group="Urine tests", min=1.000, max=1.040, step=0.005, fmt="%.3f", desc="Urine specific gravity (dataset values: 1.005-1.025)."),
    "al":   dict(label="Albumin", unit="0-5 scale", group="Urine tests", min=0.0, max=5.0, step=1.0, fmt="%.0f", desc="Urine albumin level on a 0-5 scale."),
    "su":   dict(label="Sugar", unit="0-5 scale", group="Urine tests", min=0.0, max=5.0, step=1.0, fmt="%.0f", desc="Urine sugar level on a 0-5 scale."),
    "rbc":  dict(label="Red blood cells (urine)", group="Urine tests", desc="Red blood cells in urine: normal / abnormal."),
    "pc":   dict(label="Pus cell", group="Urine tests", desc="Pus cells in urine: normal / abnormal."),
    "pcc":  dict(label="Pus cell clumps", group="Urine tests", desc="Pus cell clumps: present / notpresent."),
    "ba":   dict(label="Bacteria", group="Urine tests", desc="Bacteria in urine: present / notpresent."),
    "bgr":  dict(label="Blood glucose (random)", unit="mg/dL", group="Blood tests", min=20.0, max=700.0, step=5.0, fmt="%.0f", desc="Random blood glucose."),
    "bu":   dict(label="Blood urea", unit="mg/dL", group="Blood tests", min=1.0, max=500.0, step=1.0, fmt="%.1f", desc="Blood urea."),
    "sc":   dict(label="Serum creatinine", unit="mg/dL", group="Blood tests", min=0.1, max=100.0, step=0.1, fmt="%.2f", desc="Serum creatinine."),
    "sod":  dict(label="Sodium", unit="mEq/L", group="Blood tests", min=90.0, max=180.0, step=1.0, fmt="%.0f", desc="Serum sodium."),
    "pot":  dict(label="Potassium", unit="mEq/L", group="Blood tests", min=1.0, max=60.0, step=0.1, fmt="%.1f", desc="Serum potassium."),
    "hemo": dict(label="Hemoglobin", unit="g/dL", group="Blood tests", min=2.0, max=22.0, step=0.1, fmt="%.1f", desc="Hemoglobin."),
    "pcv":  dict(label="Packed cell volume", unit="%", group="Blood tests", min=5.0, max=70.0, step=1.0, fmt="%.0f", desc="Packed cell volume."),
    "wbcc": dict(label="White blood cell count", unit="cells/cumm", group="Blood tests", min=500.0, max=50000.0, step=100.0, fmt="%.0f", desc="White blood cell count."),
    "rbcc": dict(label="Red blood cell count", unit="millions/cmm", group="Blood tests", min=1.0, max=10.0, step=0.1, fmt="%.1f", desc="Red blood cell count."),
    "htn":  dict(label="Hypertension", group="History & symptoms", desc="History of hypertension: yes / no."),
    "dm":   dict(label="Diabetes mellitus", group="History & symptoms", desc="Diabetes mellitus: yes / no."),
    "cad":  dict(label="Coronary artery disease", group="History & symptoms", desc="Coronary artery disease: yes / no."),
    "appet": dict(label="Appetite", group="History & symptoms", desc="Appetite: good / poor."),
    "pe":   dict(label="Pedal edema", group="History & symptoms", desc="Swelling of the feet/ankles: yes / no."),
    "ane":  dict(label="Anemia", group="History & symptoms", desc="Anemia: yes / no."),
}
