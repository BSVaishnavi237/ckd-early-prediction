"""
CKD Early Prediction & Feature Selection - Streamlit dashboard.

Loads the artifacts written by train_model.py (nothing is retrained here):
    streamlit run app.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st

import ckd_common as cc

st.set_page_config(page_title="CKD Early Prediction Dashboard", page_icon="🩺", layout="wide")

DISCLAIMER = ("**Educational prototype - not a medical diagnostic tool.** This model was trained on a small, "
              "public teaching dataset (UCI CKD) for a Data Mining course project. It must not be used to "
              "diagnose, screen or make decisions about any real patient. Consult a qualified clinician.")
FRIENDLY = {"ckd": "CKD", "notckd": "Not CKD"}
COLORS = {"ckd": "#d1495b", "notckd": "#2e86ab"}
METRICS = ["Accuracy", "CKD Precision", "CKD Recall", "CKD F1"]


# --------------------------------------------------------------------------
# Small UI helpers (version-tolerant)
# --------------------------------------------------------------------------
def show_chart(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def show_table(data, **kwargs) -> None:
    try:
        st.dataframe(data, width="stretch", hide_index=True, **kwargs)
    except Exception:  # older Streamlit
        st.dataframe(data, use_container_width=True, hide_index=True, **kwargs)


def friendly(label: str) -> str:
    return FRIENDLY.get(label, label)


# --------------------------------------------------------------------------
# Loading saved artifacts (cached; invalidated only when files change)
# --------------------------------------------------------------------------
def artifact_signature() -> tuple:
    return tuple((f, (cc.ARTIFACT_DIR / f).stat().st_mtime_ns) for f in cc.REQUIRED_FILES)


@st.cache_data(show_spinner=False)
def load_tables(signature: tuple) -> dict:
    with open(cc.path("schema"), encoding="utf-8") as fh:
        schema = json.load(fh)
    with open(cc.path("summary"), encoding="utf-8") as fh:
        summary = json.load(fh)

    def typed(frame: pd.DataFrame) -> pd.DataFrame:
        for col in schema["numerical_cols"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        for col in schema["categorical_cols"]:
            frame[col] = frame[col].astype("object").where(frame[col].notna(), np.nan)
        return frame

    return {
        "schema": schema,
        "summary": summary,
        "data": typed(pd.read_csv(cc.path("cleaned"))),
        "test": typed(pd.read_csv(cc.path("test_set"))),
        "comparison": pd.read_csv(cc.path("comparison")),
        "scores": pd.read_csv(cc.path("feature_scores")),
        "cms": pd.read_csv(cc.path("confusion")),
    }


@st.cache_resource(show_spinner=False)
def load_models(signature: tuple) -> dict:
    return {
        "pipelines": joblib.load(cc.path("pipelines")),
        "label_encoder": joblib.load(cc.path("label_encoder")),
        "best_model": joblib.load(cc.path("best_model")),
    }


def load_everything():
    missing = [f for f in cc.REQUIRED_FILES if not (cc.ARTIFACT_DIR / f).exists()]
    if missing:
        st.error("Trained artifacts were not found, so the dashboard has nothing to show yet.")
        st.markdown("Missing files in `artifacts/`:\n\n" + "\n".join(f"- `{m}`" for m in missing))
        st.info("Create them once by running this in the project folder:\n\n```\npython train_model.py\n```\n"
                "Then refresh this page.")
        st.stop()
    try:
        sig = artifact_signature()
        return load_tables(sig), load_models(sig)
    except Exception as exc:  # noqa: BLE001
        st.error("The saved artifacts could not be loaded. They may be corrupted or were created with a "
                 "different scikit-learn version.")
        st.exception(exc)
        st.info("Fix: run `python train_model.py` again, then refresh this page.")
        st.stop()


# --------------------------------------------------------------------------
# Section 1 - Overview
# --------------------------------------------------------------------------
def page_overview(t: dict) -> None:
    df, schema, summary = t["data"], t["schema"], t["summary"]
    target, ckd = cc.TARGET_COL, summary["ckd_label"]
    others = [c for c in summary["class_names"] if c != ckd]
    non_ckd = others[0]
    counts = df[target].value_counts()
    features = df.drop(columns=[target])
    total_missing = int(features.isna().sum().sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Input features", len(schema["feature_columns"]))
    c3.metric("CKD records", int(counts.get(ckd, 0)))
    c4.metric("Non-CKD records", int(counts.get(non_ckd, 0)))
    c5.metric("Missing values", f"{total_missing:,}",
              help="Empty cells in the input features after converting '?' placeholders to real missing values.")
    cl = summary["cleaning"]
    st.caption(f"Source: {summary['data_source']}. Raw rows: {cl['raw_rows']}; duplicates removed: "
               f"{cl['duplicates_removed']}; rows after cleaning: {cl['rows_after_cleaning']}. "
               f"{int(features.isna().any(axis=1).sum())} of {len(df)} records have at least one missing value; "
               f"{len(schema['numerical_cols'])} numerical and {len(schema['categorical_cols'])} categorical features.")

    left, right = st.columns(2)
    with left:
        st.subheader("Target distribution")
        dist = counts.rename_axis("Class").reset_index(name="Records")
        dist["Share"] = (dist["Records"] / dist["Records"].sum() * 100).round(1)
        dist["Label"] = dist["Class"].map(friendly)
        fig = px.bar(dist, x="Label", y="Records", color="Class", color_discrete_map=COLORS,
                     text=dist.apply(lambda r: f"{int(r['Records'])} ({r['Share']}%)", axis=1))
        fig.update_layout(showlegend=False, xaxis_title=None, margin=dict(t=20))
        show_chart(fig)
    with right:
        st.subheader("Missing values by feature")
        miss = features.isna().sum()
        miss = miss[miss > 0].sort_values()
        if miss.empty:
            st.success("No missing values in the input features.")
        else:
            mdf = miss.rename_axis("Feature").reset_index(name="Missing")
            mdf["Percent"] = (mdf["Missing"] / len(df) * 100).round(1)
            fig = px.bar(mdf, x="Missing", y="Feature", orientation="h",
                         text=mdf.apply(lambda r: f"{int(r['Missing'])} ({r['Percent']}%)", axis=1))
            fig.update_layout(margin=dict(t=20), yaxis_title=None)
            show_chart(fig)
    st.caption("Missing values are filled inside the model pipeline (median for numbers, most frequent for "
               "categories), using statistics from the training split only.")

    st.subheader("Explore the clinical measurements")
    tab_num, tab_cat, tab_corr = st.tabs(["Numerical measurements", "Categorical indicators", "Correlations"])
    num_cols = schema["numerical_cols"]
    cat_cols = schema["categorical_cols"]

    with tab_num:
        default = next((c for c in ("hemo", "sc", "bgr") if c in num_cols), num_cols[0])
        col = st.selectbox("Measurement", num_cols, index=num_cols.index(default),
                           format_func=lambda c: _field_title(c))
        a, b = st.columns(2)
        with a:
            fig = px.box(df, x=target, y=col, color=target, color_discrete_map=COLORS, points="outliers")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=_field_title(col), margin=dict(t=20))
            show_chart(fig)
        with b:
            fig = px.histogram(df, x=col, color=target, color_discrete_map=COLORS, barmode="overlay",
                               opacity=0.65, nbins=30)
            fig.update_layout(xaxis_title=_field_title(col), yaxis_title="Records", margin=dict(t=20))
            show_chart(fig)
        stats = df.groupby(target)[col].describe().round(2).reset_index()
        show_table(stats)

    with tab_cat:
        if not cat_cols:
            st.info("No categorical features.")
        else:
            col = st.selectbox("Indicator", cat_cols, format_func=lambda c: _field_title(c))
            plot_df = df.assign(**{col: df[col].fillna("(missing)")})
            fig = px.histogram(plot_df, x=col, color=target, color_discrete_map=COLORS, barmode="group",
                               text_auto=True)
            fig.update_layout(xaxis_title=_field_title(col), yaxis_title="Records", margin=dict(t=20))
            show_chart(fig)

    with tab_corr:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=620, margin=dict(t=20))
        show_chart(fig)
        st.caption("Pearson correlation of the numerical attributes (pairwise, ignoring missing values).")

    with st.expander("Preview the cleaned dataset"):
        show_table(df.head(50))


def _field_title(col: str) -> str:
    meta = cc.FIELD_META.get(col)
    if not meta:
        return col
    unit = f" ({meta['unit']})" if meta.get("unit") else ""
    return f"{meta['label']} [{col}]{unit}"


# --------------------------------------------------------------------------
# Section 2 - Feature analysis
# --------------------------------------------------------------------------
def page_features(t: dict) -> None:
    scores, summary = t["scores"], t["summary"]
    k = summary["n_selected_features"]
    selected = scores[scores["Selected"]].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Encoded features scored", summary["n_encoded_features"],
              help="Numerical columns plus one-hot columns created from the categorical attributes.")
    c2.metric("Features kept (SelectKBest)", k)
    c3.metric("Training records used", summary["n_train"])

    show_all = st.toggle("Show all encoded features instead of only the selected ones", value=False)
    if show_all:
        plot_df = scores.sort_values("Mutual Information")
        fig = px.bar(plot_df, x="Mutual Information", y="Label", orientation="h", color="Selected",
                     color_discrete_map={True: "#2e86ab", False: "#b8c4cc"},
                     text=plot_df["Mutual Information"].round(3))
        fig.update_layout(height=max(450, 22 * len(plot_df)))
        st.subheader("Mutual information of every encoded feature")
    else:
        plot_df = selected.sort_values("Mutual Information")
        fig = px.bar(plot_df, x="Mutual Information", y="Label", orientation="h",
                     text=plot_df["Mutual Information"].round(3), color_discrete_sequence=["#2e86ab"])
        fig.update_layout(height=max(450, 32 * len(plot_df)))
        st.subheader(f"Top {k} features selected by mutual information")
    fig.update_layout(yaxis_title=None, margin=dict(t=20))
    show_chart(fig)
    st.caption("Scores come from the training run: mutual_info_classif(random_state=42) on the preprocessed "
               "training split. Higher = the feature tells us more about the CKD / non-CKD label.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Score table")
        show_table(scores[["Rank", "Label", "Original Attribute", "Mutual Information", "Selected"]],
                   column_config={"Mutual Information": st.column_config.NumberColumn(format="%.4f")})
        st.download_button("Download feature_scores.csv", scores.to_csv(index=False).encode("utf-8"),
                           file_name="feature_scores.csv", mime="text/csv")
    with right:
        st.subheader("Selected, grouped by original attribute")
        grouped = (selected.groupby("Original Attribute")
                   .agg(**{"Encoded features kept": ("Label", "count"),
                           "Best score": ("Mutual Information", "max")})
                   .sort_values("Best score", ascending=False).round(4).reset_index())
        show_table(grouped)

    st.subheader("Feature selection in simple language")
    st.markdown(f"""
**Why select features?** The dataset has {summary['n_features']} input attributes. Some tell us a lot about
whether a patient has CKD, others add little. Keeping only the most informative ones can make a model simpler
and easier to explain.

**What is mutual information?** Imagine guessing whether a patient has CKD. If learning one measurement
(say, hemoglobin) removes a lot of your uncertainty, that measurement has a *high* mutual-information score.
A score near 0 means knowing the value tells you almost nothing about the class.

**How it was done here**
1. Records were split into {summary['n_train']} training and {summary['n_test']} test records. Only the training records were used below.
2. Missing values were filled, numbers were scaled, and categories were turned into 0/1 columns, giving
   {summary['n_encoded_features']} encoded features. These steps learned their settings from training data only.
3. Each encoded feature received a mutual-information score against the CKD label.
4. `SelectKBest` kept the {k} highest-scoring features. The test records never influenced this choice.

**Things to keep in mind**
- Mutual information scores each feature on its own, so related measurements can all rank high even if
  they carry overlapping information.
- A categorical attribute becomes several 0/1 columns, so one attribute may appear more than once.
- High mutual information shows association in this dataset, not that a feature *causes* CKD.
- The number of features kept ({k}) is a design choice from the notebook, not a proven optimum.
""")


# --------------------------------------------------------------------------
# Section 3 - Model comparison
# --------------------------------------------------------------------------
def top_by_f1(frame: pd.DataFrame) -> str:
    best = frame["CKD F1"].max()
    tied = frame[np.isclose(frame["CKD F1"], best)]["Model"].tolist()
    if len(tied) == 1:
        return f"**{tied[0]}** has the highest CKD F1 ({best:.3f})"
    return f"**{', '.join(tied)}** are tied for the highest CKD F1 ({best:.3f})"


def page_models(t: dict) -> None:
    comp, cms, summary = t["comparison"], t["cms"], t["summary"]
    fs_full, fs_top = summary["feature_sets"]["full"], summary["feature_sets"]["selected"]
    class_names = summary["class_names"]
    n_test = summary["n_test"]
    n_pos = summary["test_class_counts"].get(summary["ckd_label"], 0)

    st.caption(f"All numbers below were computed by `train_model.py` on a held-out test set of {n_test} records "
               f"({n_pos} CKD). Precision, recall and F1 treat **CKD as the positive class**.")

    both = "Both (side by side)"
    view = st.radio("Feature set", [fs_full, fs_top, both], horizontal=True)
    shown = comp if view == both else comp[comp["Feature Set"] == view]

    st.subheader("Performance metrics")
    cfg = {m: st.column_config.NumberColumn(format="%.3f") for m in METRICS}
    show_table(shown.sort_values(["Feature Set", "CKD F1", "Accuracy"], ascending=[True, False, False]),
               column_config=cfg)
    st.download_button("Download model_comparison.csv", comp.to_csv(index=False).encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv")

    long = shown.melt(id_vars=["Model", "Feature Set"], value_vars=METRICS, var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                 facet_col="Feature Set" if view == both else None, text_auto=".3f")
    fig.update_yaxes(range=[0, 1.08])
    fig.update_layout(margin=dict(t=30))
    show_chart(fig)

    notes = []
    for fs in ([fs_full, fs_top] if view == both else [view]):
        notes.append(f"{fs}: {top_by_f1(comp[comp['Feature Set'] == fs])} on this test set.")
    st.info("  \n".join(notes) + "  \n\nThis reflects one stratified split of a small dataset: a tie, or a gap of "
            "one or two test records, is not evidence that one algorithm is truly better, and the scores do not "
            "predict performance on real-world patients.")

    st.subheader("Confusion matrices")
    cm_fs = st.radio("Confusion matrices for", [fs_full, fs_top], horizontal=True, key="cm_fs")
    models = comp["Model"].drop_duplicates().tolist()
    cols = st.columns(2)
    for i, model in enumerate(models):
        sub = cms[(cms["Model"] == model) & (cms["Feature Set"] == cm_fs)]
        mat = (sub.pivot(index="Actual", columns="Predicted", values="Count")
               .reindex(index=class_names, columns=class_names))
        fig = px.imshow(mat.values, x=[friendly(c) for c in class_names], y=[friendly(c) for c in class_names],
                        text_auto=True, color_continuous_scale="Blues", aspect="auto",
                        labels=dict(x="Predicted class", y="Actual class", color="Records"))
        fig.update_layout(title=model, coloraxis_showscale=False, margin=dict(t=50, b=10), height=320)
        with cols[i % 2]:
            show_chart(fig)

    st.subheader(f"Effect of feature selection ({fs_top} minus {fs_full})")
    wide = comp.pivot(index="Model", columns="Feature Set", values=METRICS)
    delta = pd.DataFrame({m: wide[m][fs_top] - wide[m][fs_full] for m in METRICS}).reset_index()
    delta.columns = ["Model"] + [f"Δ {m}" for m in METRICS]
    show_table(delta, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in delta.columns[1:]})
    st.caption("Positive = better with selected features, negative = worse, 0 = no change.")


# --------------------------------------------------------------------------
# Section 4 - Prediction demo
# --------------------------------------------------------------------------
MAX_UNKNOWN_FRACTION = 0.5


def bounds(col: str, schema: dict) -> tuple[float, float]:
    meta = cc.FIELD_META.get(col, {})
    if "min" in meta and "max" in meta:
        return float(meta["min"]), float(meta["max"])
    s = schema["numeric_stats"][col]
    span = max(s["max"] - s["min"], 1.0)
    return s["min"] - span, s["max"] + span


def step_and_fmt(col: str) -> tuple[float, str]:
    meta = cc.FIELD_META.get(col, {})
    return float(meta.get("step", 1.0)), meta.get("fmt", "%.2f")


def default_numeric(col: str, schema: dict) -> float:
    lo, hi = bounds(col, schema)
    return float(min(max(schema["numeric_stats"][col]["median"], lo), hi))


def default_categorical(col: str, schema: dict) -> str:
    mode = schema["categorical_mode"].get(col)
    return mode if mode is not None else schema["categorical_options"][col][0]


def reset_form(schema: dict) -> None:
    for col in schema["numerical_cols"]:
        st.session_state[f"num_{col}"] = default_numeric(col, schema)
        st.session_state[f"unk_{col}"] = False
    for col in schema["categorical_cols"]:
        st.session_state[f"cat_{col}"] = default_categorical(col, schema)
    st.session_state.pop("loaded_truth", None)


def load_test_record(test_df: pd.DataFrame, schema: dict) -> None:
    row = test_df.sample(1).iloc[0]
    for col in schema["numerical_cols"]:
        value = row[col]
        lo, hi = bounds(col, schema)
        if pd.isna(value):
            st.session_state[f"unk_{col}"] = True
            st.session_state[f"num_{col}"] = default_numeric(col, schema)
        else:
            st.session_state[f"unk_{col}"] = False
            st.session_state[f"num_{col}"] = float(min(max(float(value), lo), hi))
    for col in schema["categorical_cols"]:
        value = row[col]
        st.session_state[f"cat_{col}"] = "(not available)" if pd.isna(value) else value
    st.session_state["loaded_truth"] = row[cc.TARGET_COL]


def page_prediction(t: dict, m: dict) -> None:
    schema, summary = t["schema"], t["summary"]
    class_names, ckd_idx = summary["class_names"], summary["ckd_index"]
    fs_full, fs_top = summary["feature_sets"]["full"], summary["feature_sets"]["selected"]
    pipelines, label_encoder = m["pipelines"], m["label_encoder"]

    st.warning(DISCLAIMER)

    for col in schema["numerical_cols"]:
        st.session_state.setdefault(f"num_{col}", default_numeric(col, schema))
        st.session_state.setdefault(f"unk_{col}", False)
    for col in schema["categorical_cols"]:
        st.session_state.setdefault(f"cat_{col}", default_categorical(col, schema))

    # ---- Model choice
    best = summary["best_saved"]
    model_names = list(pipelines[fs_full].keys())
    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Model", model_names, index=model_names.index(best["model"]),
                              help=f"Default = the model saved in ckd_model.pkl ({best['model']}), chosen as: {best['rule']}.")
    feature_set = c2.radio("Feature set", [fs_full, fs_top], horizontal=True,
                           help="The pipeline for this choice includes its own fitted preprocessing "
                                "(and feature selection, if applicable).")
    pipe = pipelines[feature_set][model_name]

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load a random test record", on_click=load_test_record, args=(t["test"], schema),
              help="Fills the form with a real record from the held-out test set.")
    b2.button("Reset to typical values", on_click=reset_form, args=(schema,))
    if "loaded_truth" in st.session_state:
        st.caption(f"A test record is loaded. Its label in the dataset is **{st.session_state['loaded_truth']}** "
                   "(shown for comparison only; it no longer applies if you edit the values).")
    st.caption("Defaults are typical training values (median for numbers, most frequent for categories). "
               "Tick *not available* / choose *(not available)* for anything you do not know - it will be "
               "handled exactly like missing data during training.")

    # ---- Form: same column names/order/dtypes as the training data
    groups = {g: [] for g in cc.GROUP_ORDER}
    groups["Other"] = []
    for col in schema["feature_columns"]:
        groups.setdefault(cc.FIELD_META.get(col, {}).get("group", "Other"), []).append(col)

    with st.form("prediction_form"):
        for group, cols_in_group in groups.items():
            if not cols_in_group:
                continue
            st.markdown(f"**{group}**")
            grid = st.columns(3)
            for i, col in enumerate(cols_in_group):
                meta = cc.FIELD_META.get(col, {})
                label = meta.get("label", col) + (f" ({meta['unit']})" if meta.get("unit") else "") + f" [{col}]"
                with grid[i % 3]:
                    if col in schema["numerical_cols"]:
                        lo, hi = bounds(col, schema)
                        step, fmt = step_and_fmt(col)
                        st.number_input(label, min_value=lo, max_value=hi, step=step, format=fmt,
                                        key=f"num_{col}", help=meta.get("desc"))
                        st.checkbox("not available", key=f"unk_{col}")
                    else:
                        options = ["(not available)"] + schema["categorical_options"][col]
                        st.selectbox(label, options, key=f"cat_{col}", help=meta.get("desc"))
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    # ---- Build the input row exactly like the training data
    row, unknown, out_of_range = {}, [], []
    for col in schema["feature_columns"]:
        if col in schema["numerical_cols"]:
            if st.session_state[f"unk_{col}"]:
                row[col] = np.nan
                unknown.append(col)
            else:
                value = float(st.session_state[f"num_{col}"])
                row[col] = value
                s = schema["numeric_stats"][col]
                if value < s["min"] or value > s["max"]:
                    out_of_range.append(f"{col} = {value:g} (training range {s['min']:g} to {s['max']:g})")
        else:
            value = st.session_state[f"cat_{col}"]
            if value == "(not available)":
                row[col] = np.nan
                unknown.append(col)
            elif value not in schema["categorical_options"][col]:
                st.error(f"Invalid value '{value}' for {col}. Allowed: {schema['categorical_options'][col]}")
                return
            else:
                row[col] = value

    if len(unknown) / len(schema["feature_columns"]) > MAX_UNKNOWN_FRACTION:
        st.error(f"{len(unknown)} of {len(schema['feature_columns'])} attributes are marked unavailable. "
                 f"Please provide at least {int((1 - MAX_UNKNOWN_FRACTION) * len(schema['feature_columns']))} "
                 "values - a prediction based mostly on imputed data would be meaningless.")
        return

    X_input = pd.DataFrame([row], columns=schema["feature_columns"])
    for col in schema["numerical_cols"]:
        X_input[col] = X_input[col].astype("float64")
    for col in schema["categorical_cols"]:
        X_input[col] = X_input[col].astype("object")

    try:
        pred = int(pipe.predict(X_input)[0])
        proba = None
        if hasattr(pipe, "predict_proba"):
            proba = float(pipe.predict_proba(X_input)[0][list(pipe.classes_).index(ckd_idx)])
    except Exception as exc:  # noqa: BLE001
        st.error("The model could not make a prediction for these inputs.")
        st.exception(exc)
        return

    pred_label = label_encoder.inverse_transform([pred])[0]
    st.subheader("Result")
    r1, r2 = st.columns(2)
    r1.metric("Predicted class", f"{friendly(pred_label)}  ({pred_label})")
    if proba is not None:
        r2.metric("Model's estimated probability of CKD", f"{proba:.1%}",
                  help="Raw model output. It is not a calibrated clinical risk.")
    else:
        r2.caption(f"{model_name} (as configured) does not output probabilities; only the class is shown.")
    st.caption(f"Model: {model_name} | Feature set: {feature_set}")

    if unknown:
        st.info("Treated as missing and filled by the pipeline's training-set imputers: " + ", ".join(unknown))
    if out_of_range:
        st.warning("Outside the range seen in the training data (the prediction may be unreliable): "
                   + "; ".join(out_of_range))
    with st.expander("Exact input row sent to the pipeline"):
        show_table(X_input)
    st.warning(DISCLAIMER)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("🩺 Early Prediction and Feature Selection of Chronic Kidney Disease")
    tables, models = load_everything()
    summary = tables["summary"]

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Section", ["Overview", "Feature Analysis", "Model Comparison", "Prediction Demo"],
                            label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Trained: {summary['created_utc']}  \nSource: {summary['data_source']}  \n"
                       f"Split: {summary['n_train']} train / {summary['n_test']} test, random_state={summary['random_state']}")
    trained_with = summary["versions"]["scikit_learn"]
    if trained_with != sklearn.__version__:
        st.sidebar.warning(f"Artifacts were trained with scikit-learn {trained_with}, you are running "
                           f"{sklearn.__version__}. If anything fails, rerun `python train_model.py`.")
    st.sidebar.info("Educational prototype - not a medical diagnostic tool.")

    if page == "Overview":
        page_overview(tables)
    elif page == "Feature Analysis":
        page_features(tables)
    elif page == "Model Comparison":
        page_models(tables)
    else:
        page_prediction(tables, models)


main()
