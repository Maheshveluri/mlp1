"""Streamlit prediction interface for the trained solar Random Forest."""
from pathlib import Path

import joblib
import json
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.tree import export_text, plot_tree

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "solar_model.joblib"
ACTUAL_LOOKUP_PATH = ARTIFACTS / "actual_efficiency_lookup.csv"

st.set_page_config(page_title="Solar Cell Efficiency Predictor", page_icon="☀️", layout="wide")
st.title("☀️ Solar Cell Efficiency Prediction")
st.caption("Random Forest regression trained on the supplied solar cell dataset. Predicted PCE is shown as a percentage.")

summary_path = ARTIFACTS / "dataset_summary.json"
if summary_path.exists():
    with st.expander("Dataset inspection"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        st.write("Workbook shape:", summary["shape_before_cleaning"])
        st.write("Columns:", summary["columns"])
        st.write("Data types:", summary["data_types"])
        st.write("Duplicate records:", summary["duplicate_records"])
        st.write("Missing values by column:", summary["missing_values"])
        st.dataframe(pd.DataFrame(summary["first_five_rows"]), width="stretch")

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python train_model.py --data \"Solar Cell data set.csv.xlsx\"` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
# Avoid spawning prediction worker processes in restricted Windows environments.
pipeline.named_steps["model"].set_params(n_jobs=1)
forest = pipeline.named_steps["model"]
features = bundle["feature_columns"]
numeric = set(bundle["numeric_features"])
categorical = set(bundle["categorical_features"])
importance_path = ARTIFACTS / "feature_importance.csv"
if importance_path.exists():
    importance_df = pd.read_csv(importance_path)
    input_features = importance_df.head(10)["feature"].tolist()
else:
    importance_df = pd.DataFrame(columns=["feature", "importance"])
    input_features = features[:10]


@st.cache_data(show_spinner=False)
def load_actual_lookup(path: str):
    return pd.read_csv(path)


def find_recorded_actuals(input_row):
    """Return measured PCE values for exact matches across all entered features."""
    if not ACTUAL_LOOKUP_PATH.exists():
        return pd.Series(dtype=float)
    if any(pd.isna(input_row.get(feature)) for feature in input_features):
        return pd.Series(dtype=float)
    lookup = load_actual_lookup(str(ACTUAL_LOOKUP_PATH))
    mask = pd.Series(True, index=lookup.index)
    for feature in input_features:
        value = input_row[feature]
        if feature in numeric:
            column = pd.to_numeric(lookup[feature], errors="coerce")
            mask &= column.eq(float(value))
        else:
            column = lookup[feature].map(lambda item: str(item).strip() if pd.notna(item) else "")
            mask &= column.eq(str(value).strip())
    return pd.to_numeric(lookup.loc[mask, bundle["target"]], errors="coerce").dropna()

with st.sidebar:
    st.header("Model results")
    metrics_path = ARTIFACTS / "metrics.json"
    if metrics_path.exists():
        metrics = pd.read_json(metrics_path, typ="series")
        for key, label in [("r2", "R²"), ("mae", "MAE (percentage points)"),
                           ("mse", "MSE"), ("rmse", "RMSE (percentage points)")]:
            if key in metrics:
                st.metric(label, f"{float(metrics[key]):.4f}")
        st.caption(f"Held-out test rows: {int(metrics.get('test_rows', 0)):,}")
    st.info("R² measures variance explained. MAE is the average absolute error in percentage points. MSE averages squared errors, so large errors count more. RMSE is the square root of MSE and is in percentage points.")

st.subheader("Enter the 10 most important solar cell features")
st.write("These are the ten highest-ranked features and the complete input schema used to train the saved model. Blank fields use the training-data imputation values.")
st.caption("For Perovskite_deposition_quenching_induced_crystallisation, enter 1 for True or 0 for False. Click Predict Efficiency; the result appears below the form.")

tree_col, depth_col = st.columns(2)
with tree_col:
    tree_index = st.number_input(
        "Tree to trace (starting at 1)", min_value=1,
        max_value=len(forest.estimators_), value=1, step=1,
    )
with depth_col:
    display_depth = st.slider(
        "Diagram depth to display", min_value=1, max_value=6, value=3,
        help="The full fitted trees can be very large. This limits the displayed diagram depth.",
    )

with st.form("prediction_form"):
    input_values = {}
    cols = st.columns(2)
    for i, feature in enumerate(input_features):
        with cols[i % 2]:
            if feature in numeric:
                val = st.text_input(feature, value="", placeholder="Leave blank for training median", key=f"num_{feature}")
                input_values[feature] = val
            elif feature in categorical:
                # Use the training dataset's common values when available. A text field
                # supports additional categories; unknown values are handled by the encoder.
                value = st.text_input(feature, value="", placeholder="Enter a category or leave blank", key=f"cat_{feature}")
                input_values[feature] = value
            else:
                input_values[feature] = st.text_input(feature, value="", key=f"other_{feature}")
    submitted = st.form_submit_button("Predict Efficiency", type="primary")

if submitted:
    # Keep the saved ten-feature schema; blank fields pass through the fitted imputers.
    row = {feature: float("nan") for feature in features}
    errors = []
    for feature in input_features:
        value = input_values[feature]
        if feature in numeric:
            if not value.strip():
                row[feature] = float("nan")
            else:
                try:
                    number = float(value)
                    if number != number or number in (float("inf"), float("-inf")):
                        raise ValueError
                    row[feature] = number
                except ValueError:
                    errors.append(f"{feature} must be a valid finite number.")
        else:
            row[feature] = value.strip() or None
    if errors:
        for error in errors:
            st.error(error)
    else:
        try:
            sample = pd.DataFrame([row], columns=features)
            prediction = float(pipeline.predict(sample)[0])
            actuals = find_recorded_actuals(row)
            chosen_tree = forest.estimators_[int(tree_index) - 1]
            transformed_sample = pipeline.named_steps["preprocess"].transform(sample)
            path_nodes = chosen_tree.decision_path(transformed_sample).indices
            tree_structure = chosen_tree.tree_
            transformed_feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
            if actuals.empty:
                actual_label = "Not available: no exact dataset match"
            elif actuals.nunique() == 1:
                actual_label = f"{float(actuals.iloc[0]):.2f}% (recorded)"
            else:
                actual_label = f"Multiple values, {actuals.min():.2f}%–{actuals.max():.2f}%"

            trace_rows = []
            for stage, node_id in enumerate(path_nodes, start=1):
                node_feature_idx = int(tree_structure.feature[node_id])
                node_pce = float(tree_structure.value[node_id].ravel()[0])
                if node_feature_idx < 0:
                    split_feature = "Leaf (final node)"
                    encoded_value = "—"
                    rule = "No further split"
                    branch = "Stop"
                else:
                    split_feature = str(transformed_feature_names[node_feature_idx])
                    encoded_value = float(transformed_sample[0, node_feature_idx])
                    threshold = float(tree_structure.threshold[node_id])
                    went_left = encoded_value <= threshold
                    rule = f"{split_feature} {'<=' if went_left else '>'} {threshold:.3f}"
                    branch = "Left" if went_left else "Right"
                trace_rows.append({
                    "Stage": stage,
                    "Split / node": split_feature,
                    "Input at split (encoded)": encoded_value,
                    "Rule evaluated": rule,
                    "Branch taken": branch,
                    "Tree estimate at this node (%)": node_pce,
                    "Recorded actual PCE": actual_label,
                })

            predicted_col, actual_col = st.columns(2)
            predicted_col.metric("Predicted PCE", f"{prediction:.2f}%")
            actual_col.metric("Recorded actual PCE", actual_label)
            if actuals.empty:
                st.info("No exact row matches all ten entered values. For a new cell, actual efficiency is known only after it is measured.")
            elif actuals.nunique() == 1:
                if len(actuals) > 1:
                    st.caption(f"Matched {len(actuals)} identical dataset rows.")
            else:
                st.info(f"Found {len(actuals)} dataset rows with these ten values. Their mean recorded PCE is {actuals.mean():.2f}%; range: {actuals.min():.2f}%–{actuals.max():.2f}%.")
                with st.expander("Matching dataset actual values"):
                    st.dataframe(actuals.rename("Recorded actual PCE (%)").reset_index(drop=True), width="stretch")

            st.subheader(f"Input path through Random Forest tree {int(tree_index)}")
            tree_prediction = float(chosen_tree.predict(transformed_sample)[0])
            st.write(
                f"Selected tree estimate: **{tree_prediction:.2f}%**. "
                f"Forest prediction (average of {len(forest.estimators_)} trees): **{prediction:.2f}%**."
            )
            st.caption(
                "Each row shows the PCE estimate stored at that node and the branch taken. "
                "The recorded actual is a single measured target for a matching complete input; "
                "it cannot be separately measured at each split. Encoded values are the values "
                "the fitted preprocessing step passes into the tree."
            )
            st.dataframe(pd.DataFrame(trace_rows), hide_index=True, width="stretch")
        except Exception as exc:
            st.error(f"Could not make a prediction: {exc}")

st.divider()
st.subheader("Evaluation and visualizations")
metrics_path = ARTIFACTS / "metrics.json"
if metrics_path.exists():
    metrics = pd.read_json(metrics_path, typ="series")
    st.write(f"Test set: {int(metrics.get('test_rows', 0)):,} unseen examples; train/test split is 80:20 (random state {int(metrics.get('random_state', 42))}).")
    a, b, c, d = st.columns(4)
    a.metric("R²", f"{float(metrics['r2']):.4f}")
    b.metric("MAE", f"{float(metrics['mae']):.4f} pp")
    c.metric("MSE", f"{float(metrics['mse']):.4f}")
    d.metric("RMSE", f"{float(metrics['rmse']):.4f} pp")
    st.caption("R² shows how much of the test-set variation the model explains (1 is ideal; it can be negative). MAE gives the average absolute miss in efficiency percentage points. MSE squares errors and penalizes large misses more. RMSE is the square root of MSE, expressed in percentage points.")

pred_path = ARTIFACTS / "test_predictions.csv"
if pred_path.exists():
    with st.expander("Actual versus predicted test values"):
        st.dataframe(pd.read_csv(pred_path).head(1000), width="stretch")

for image_name, caption in [
    ("feature_distributions.png", "Numeric feature distributions"),
    ("correlation_heatmap.png", "Correlation among numeric features and target"),
    ("actual_vs_predicted.png", "Actual versus predicted efficiency"),
]:
    image_path = ARTIFACTS / image_name
    if image_path.exists():
        st.image(str(image_path), caption=caption, width="stretch")

if not importance_df.empty:
    st.subheader("Top 10 feature importance")
    top10 = importance_df.head(10).sort_values("importance", ascending=True)
    st.bar_chart(top10.set_index("feature"), horizontal=True, x_label="Importance", width="stretch")
    with st.expander("Top 10 importance values"):
        st.dataframe(importance_df.head(10), width="stretch")

st.subheader("Random Forest tree structure")
st.write(
    f"The Random Forest contains {len(forest.estimators_)} decision trees. "
    "Each tree makes a sequence of feature-based splits; the forest averages "
    "their outputs to predict PCE. Explore one fitted tree below."
)
try:
    transformed_feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    chosen_tree = forest.estimators_[int(tree_index) - 1]
    # Shorten encoded feature names in the diagram; the complete names remain in the
    # rule text below. This keeps the split labels legible without changing the tree.
    display_feature_names = []
    for name in transformed_feature_names:
        short_name = str(name).replace("categorical__", "").replace("numeric__", "")
        display_feature_names.append(short_name if len(short_name) <= 38 else short_name[:35] + "…")
    fig, ax = plt.subplots(figsize=(30, 16))
    plot_tree(
        chosen_tree, feature_names=display_feature_names, filled=True,
        rounded=True, max_depth=display_depth, precision=2, ax=ax,
        fontsize=10,
    )
    ax.set_title(f"Random Forest tree {int(tree_index)} (displayed through depth {display_depth})")
    fig.tight_layout()
    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)
    st.pyplot(fig, clear_figure=True)
    st.download_button(
        "Download this tree diagram as PDF",
        data=pdf_buffer.getvalue(),
        file_name=f"random_forest_tree_{int(tree_index)}_depth_{display_depth}.pdf",
        mime="application/pdf",
    )
    plt.close(fig)
    tree_text = export_text(
        chosen_tree, feature_names=list(transformed_feature_names),
        max_depth=display_depth,
    )
    with st.expander("View this tree's split rules as text"):
        st.code(tree_text, language="text")
    st.caption(
        "Split labels may include one-hot encoded categories. The displayed depth is "
        "truncated for readability; the saved model retains the complete forest."
    )
except Exception as exc:
    st.error(f"Could not display the fitted tree: {exc}")
