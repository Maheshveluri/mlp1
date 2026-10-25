# Solar Cell Efficiency Prediction using Random Forest
# Simple version: edit DATA_FILE if your Excel/CSV file is stored elsewhere.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
DATA_FILE = "Solar Cell data set.csv.xlsx"  # keep dataset in the same folder as this code
TARGET = "JV_default_PCE"
OUTPUT_DIR = "solar_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DATA_FILE.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst five rows:")
print(df.head())

# 2. Basic cleaning
df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])
print("\nShape after removing duplicates / missing target:", df.shape)

# IMPORTANT:
# These JV values are measured alongside PCE. Excluding them helps predict
# efficiency from cell fabrication/design details before electrical testing.
leakage_columns = ["JV_default_Voc", "JV_default_Jsc", "JV_default_FF", "JV_measured"]
X = df.drop(columns=[TARGET] + leakage_columns, errors="ignore")
y = df[TARGET]

# 3. Create simple EDA plots
sns.set_theme(style="whitegrid")

# Target distribution
plt.figure(figsize=(8, 5))
sns.histplot(y, bins=35, kde=True)
plt.title("Distribution of Solar Cell Efficiency (PCE)")
plt.xlabel("PCE (%)")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_pce_distribution.png", dpi=160)
plt.show()

# Missing values
missing = df.isna().mean().mul(100).sort_values(ascending=False)
missing = missing[missing > 0]
if len(missing):
    plt.figure(figsize=(10, 6))
    missing.head(20).sort_values().plot(kind="barh")
    plt.title("Top 20 Columns with Missing Values")
    plt.xlabel("Missing values (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_missing_values.png", dpi=160)
    plt.show()
else:
    print("No missing values found.")

# Numeric correlation (EDA only)
numeric_df = df.select_dtypes(include=np.number)
if numeric_df.shape[1] > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Correlation Between Numeric Columns")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_numeric_correlation.png", dpi=160)
    plt.show()

# 4. Identify numeric and categorical input columns
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Handle true/false and mixed categorical values consistently
for col in categorical_features:
    X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)

numeric_pipeline = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="median"))
])
categorical_pipeline = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
])
preprocessor = ColumnTransformer(transformers=[
    ("numbers", numeric_pipeline, numeric_features),
    ("categories", categorical_pipeline, categorical_features)
])

# 5. Split data and train Random Forest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = Pipeline(steps=[
    ("prepare", preprocessor),
    ("random_forest", RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 6. Compare model predictions with actual values
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print("\nMODEL RESULTS")
print(f"R-squared: {r2:.3f}")
print(f"MAE:       {mae:.3f} PCE percentage points")
print(f"RMSE:      {rmse:.3f} PCE percentage points")

# Actual vs predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.35)
low = min(y_test.min(), predictions.min())
high = max(y_test.max(), predictions.max())
plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
plt.xlabel("Actual PCE (%)")
plt.ylabel("Predicted PCE (%)")
plt.title("Actual vs Predicted Solar Cell Efficiency")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_actual_vs_predicted.png", dpi=160)
plt.show()

# Residuals: errors should be scattered around zero
residuals = y_test - predictions
plt.figure(figsize=(8, 5))
sns.scatterplot(x=predictions, y=residuals, alpha=0.35)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted PCE (%)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_residuals.png", dpi=160)
plt.show()

# Error distribution
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=35, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Prediction error (PCE percentage points)")
plt.title("Distribution of Prediction Errors")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_error_distribution.png", dpi=160)
plt.show()

# 7. Feature importance from Random Forest
rf = model.named_steps["random_forest"]
feature_names = model.named_steps["prepare"].get_feature_names_out()
importance = pd.Series(rf.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=False).head(20).sort_values()
plt.figure(figsize=(10, 7))
importance.plot(kind="barh")
plt.xlabel("Random Forest importance")
plt.title("Top 20 Important Input Features")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_feature_importance.png", dpi=160)
plt.show()

# 8. Save test predictions and model
comparison = pd.DataFrame({
    "Actual_PCE": y_test.to_numpy(),
    "Predicted_PCE": predictions,
    "Error": residuals.to_numpy()
})
comparison.to_csv(f"{OUTPUT_DIR}/test_predictions.csv", index=False)

import joblib
joblib.dump(model, f"{OUTPUT_DIR}/solar_random_forest.joblib")
print(f"\nDone. Plots, predictions and model are saved in: {OUTPUT_DIR}")

# 9. Predict a new solar cell record
# Replace these example values with the fabrication/design details of your cell.
# Use exact column names from the dataset. Unentered columns are imputed.
new_cell = pd.DataFrame([{
    "Cell_architecture": "n-i-p",
    "Perovskite_deposition_number_of_deposition_steps": 1,
    "Encapsulation": True
}]).reindex(columns=X.columns)

new_prediction = model.predict(new_cell)[0]
print(f"\\nEstimated PCE for the entered new cell: {new_prediction:.2f}%")
print("This is an estimate; reliability depends on similarity to the training data.")
