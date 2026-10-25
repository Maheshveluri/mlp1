"""Train and evaluate a Random Forest regressor for the supplied solar dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "JV_default_PCE"
OUTPUT_COLUMNS = ["JV_measured", "JV_default_Voc", "JV_default_Jsc", "JV_default_FF"]
RANDOM_STATE = 42


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Use an Excel (.xlsx/.xls) or CSV dataset.")


def analyze_and_clean(raw: pd.DataFrame):
    if TARGET not in raw.columns:
        raise ValueError(f"Expected target column {TARGET!r}. Found: {list(raw.columns)}")
    summary = {
        "shape_before_cleaning": list(raw.shape),
        "columns": list(raw.columns),
        "data_types": {str(k): str(v) for k, v in raw.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in raw.isna().sum().items()},
        "duplicate_records": int(raw.duplicated().sum()),
        "first_five_rows": raw.head().replace({np.nan: None}).to_dict(orient="records"),
        "target": TARGET,
        "units": "% (PCE)",
        "excluded_measurement_columns": [c for c in OUTPUT_COLUMNS if c in raw.columns],
    }
    df = raw.drop_duplicates().copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")
    # Efficiency is a percentage. Discard missing, non-finite, and out-of-range labels.
    valid_y = y.notna() & np.isfinite(y) & y.between(0, 100)
    df = df.loc[valid_y].copy()
    y = y.loc[valid_y].astype(float)
    excluded = [TARGET] + [c for c in OUTPUT_COLUMNS if c in df.columns]
    X = df.drop(columns=excluded).copy()

    # Parse numeric-like object columns without forcing genuinely categorical strings
    # into numbers. Treat +/- infinity as missing and impute downstream in the pipeline.
    numeric_cols, categorical_cols = [], []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_bool_dtype(s.dtype) or pd.api.types.is_numeric_dtype(s.dtype):
            numeric_cols.append(col)
            X[col] = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            converted = pd.to_numeric(s, errors="coerce")
            present = s.notna().sum()
            ratio = converted.notna().sum() / present if present else 0
            if present and ratio >= 0.90:
                numeric_cols.append(col)
                X[col] = converted.replace([np.inf, -np.inf], np.nan)
            else:
                categorical_cols.append(col)
                X[col] = s.map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else np.nan)
    summary.update({
        "shape_after_cleaning": [int(len(X)), int(X.shape[1])],
        "rows_removed_as_duplicates": int(raw.duplicated().sum()),
        "rows_removed_for_invalid_or_missing_target": int(len(raw.drop_duplicates()) - len(X)),
        "feature_columns": list(X.columns),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    })
    return X, y, numeric_cols, categorical_cols, summary


def make_pipeline(numeric_cols, categorical_cols, params=None):
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical_cols))
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    model_params = {"n_estimators": 120, "n_jobs": -1, "random_state": RANDOM_STATE}
    model_params.update(params or {})
    model = RandomForestRegressor(**model_params)
    return Pipeline([("preprocess", prep), ("model", model)])


def aggregate_feature_importance(model: Pipeline, feature_columns):
    """Sum one-hot encoded importances back to their original input columns."""
    # Feature importance aggregation can also parallelize over trees; keep this safe
    # in restricted Windows processes.
    model.named_steps["model"].set_params(n_jobs=1)
    names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_
    grouped = {column: 0.0 for column in feature_columns}
    for name, importance in zip(names, importances):
        raw_name = name.split("__", 1)[-1]
        source = next((column for column in feature_columns
                       if raw_name == column or raw_name.startswith(column + "_")), None)
        if source is not None:
            grouped[source] += float(importance)
    return pd.DataFrame(sorted(grouped.items(), key=lambda item: item[1], reverse=True),
                        columns=["feature", "importance"])


def make_visualizations(raw: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                        y_test: pd.Series, predictions: np.ndarray, model: Pipeline, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    # Distributions for a selection of usable numeric fields (and target).
    numeric = [c for c in X.columns
               if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    selected = numeric[:8]
    if selected:
        distribution_data = X[selected].copy()
        for column in selected:
            if pd.api.types.is_bool_dtype(distribution_data[column]):
                distribution_data[column] = distribution_data[column].astype("int8")
        distribution_data.hist(figsize=(12, 8), bins=30, layout=(-1, 3))
        plt.suptitle("Numeric feature distributions")
        plt.tight_layout()
        plt.savefig(out / "feature_distributions.png", dpi=150)
        plt.close("all")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        y.hist(ax=ax, bins=30)
        ax.set(xlabel="PCE (%)", ylabel="Row count", title="Target efficiency distribution")
        fig.tight_layout()
        fig.savefig(out / "feature_distributions.png", dpi=150)
        plt.close(fig)
    corr_df = X[numeric].copy()
    for column in numeric:
        if pd.api.types.is_bool_dtype(corr_df[column]):
            corr_df[column] = corr_df[column].astype("int8")
    corr_df[TARGET] = y
    corr = corr_df.corr(numeric_only=True)
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=7)
        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Numeric feature correlation")
        fig.tight_layout()
        fig.savefig(out / "correlation_heatmap.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, predictions, alpha=.25, s=12)
    lo, hi = min(float(y_test.min()), float(predictions.min())), max(float(y_test.max()), float(predictions.max()))
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set(xlabel="Actual PCE (%)", ylabel="Predicted PCE (%)", title="Actual vs predicted efficiency")
    fig.tight_layout()
    fig.savefig(out / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    # Aggregate impurity importance from encoded columns back to each original feature.
    importance_df = aggregate_feature_importance(model, X.columns)
    importance_df.to_csv(out / "feature_importance.csv", index=False)
    top = importance_df.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * .3)))
    ax.barh(top.feature, top.importance)
    ax.set(xlabel="Aggregated Random Forest impurity importance", title="Feature importance (top 20)")
    fig.tight_layout()
    fig.savefig(out / "feature_importance.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Solar Cell data set.csv.xlsx", help="Path to .xlsx, .xls, or .csv data")
    parser.add_argument("--out", default="artifacts", help="Output folder")
    args = parser.parse_args()
    out = Path(args.out)
    raw = read_dataset(Path(args.data))
    X, y, numeric_cols, categorical_cols, summary = analyze_and_clean(raw)
    print("Dataset shape:", raw.shape)
    print("Column names:", list(raw.columns))
    print("Data types:\n", raw.dtypes.to_string())
    print("First five rows:\n", raw.head().to_string(index=False))
    print("Missing values:\n", raw.isna().sum().to_string())
    print("Duplicate records:", int(raw.duplicated().sum()))
    print(f"Target: {TARGET} (%); features: {len(X.columns)} (measurement outputs excluded)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Select features on one training-only subset so the CV tuning subset remains
    # independent of feature selection. The test set remains untouched throughout.
    X_select, X_tune, y_select, y_tune = train_test_split(
        X_train, y_train, test_size=0.5, random_state=RANDOM_STATE
    )
    selector = make_pipeline(numeric_cols, categorical_cols)
    selector.fit(X_select, y_select)
    ranked = aggregate_feature_importance(selector, X.columns)
    selected_features = ranked.head(10)["feature"].tolist()
    selected_numeric = [c for c in numeric_cols if c in selected_features]
    selected_categorical = [c for c in categorical_cols if c in selected_features]
    summary["selected_top_10_features"] = selected_features

    # Tune the final ten-feature model using only the separate tuning subset.
    base = make_pipeline(selected_numeric, selected_categorical)
    grid = [
        {"model__n_estimators": [100], "model__max_depth": [None], "model__min_samples_split": [2], "model__min_samples_leaf": [1]},
        {"model__n_estimators": [100], "model__max_depth": [24], "model__min_samples_split": [2], "model__min_samples_leaf": [1]},
        {"model__n_estimators": [100], "model__max_depth": [24], "model__min_samples_split": [5], "model__min_samples_leaf": [2]},
        {"model__n_estimators": [160], "model__max_depth": [None], "model__min_samples_split": [5], "model__min_samples_leaf": [2]},
    ]
    search = GridSearchCV(base, grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=1, verbose=1)
    search.fit(X_tune[selected_features], y_tune)
    best_params = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
    model = make_pipeline(selected_numeric, selected_categorical, params=best_params)
    model.fit(X_train[selected_features], y_train)
    model.named_steps["model"].set_params(n_jobs=1)
    predictions = model.predict(X_test[selected_features])
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "target": TARGET, "units": "%", "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "features": len(selected_features), "selected_features": selected_features,
        "best_parameters": search.best_params_,
        "cv_best_mae": float(-search.best_score_), "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)), "mse": float(mse), "rmse": float(np.sqrt(mse)),
    }
    summary["evaluation"] = metrics
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"pipeline": model, "feature_columns": selected_features,
                 "numeric_features": selected_numeric, "categorical_features": selected_categorical,
                 "target": TARGET, "units": "%", "metrics": metrics},
                out / "solar_model.joblib", compress=3)
    # Keep labeled source rows for an honest actual-versus-predicted lookup in the app.
    actual_lookup = X[selected_features].copy()
    actual_lookup[TARGET] = y
    actual_lookup.to_csv(out / "actual_efficiency_lookup.csv", index=False)
    result = pd.DataFrame({"actual_efficiency_percent": y_test, "predicted_efficiency_percent": predictions})
    result.to_csv(out / "test_predictions.csv", index_label="row_index")
    make_visualizations(raw, X[selected_features], y, y_test, predictions, model, out)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model, predictions, and plots to {out.resolve()}")


if __name__ == "__main__":
    main()
