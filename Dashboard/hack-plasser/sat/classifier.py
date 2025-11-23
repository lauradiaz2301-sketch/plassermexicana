import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

#df = pd.read_csv("../data/sat.csv")
#features = [
#    "NDVI", "NDWI", "NDMI", "NBR", "clay_frac", "mm_interval", "mm_cumulative",
#    "VV_bit", "VH_bit", "B_bit", "score_NBR", "score_NDWI", "score_NDMI", "score_NDVI",
#    "score_clay", "score_amplitude",
#]
#corrs = df[features + ["is_bad_track"]].corr(numeric_only=True)["is_bad_track"].sort_values()
#print(corrs)

# =====================================================
# 1. Load Data
# =====================================================
#df = pd.read_csv('../data/sat.csv')
df = pd.read_csv("../data/synthetic_timeseries_with_priority_info.csv")

# EXACT features for model
features = [
    "NDVI",
    "NDWI",
    "NDMI",
    "NBR",
    "clay_frac",
    "mm_interval",
    "mm_cumulative",
    "VV_bit",
    "VH_bit",
    "B_bit",
    "score_NBR",
    "score_NDWI",
    "score_NDMI",
    "score_NDVI",
    "score_clay",
    "score_amplitude",
]

# Features (use only good ones)
features = [
    "NDVI",
    "NDWI",
    "NDMI",
    "NBR",
    "clay_frac",
    "mm_interval",
    "mm_cumulative",
    "VV_bit",
    "VH_bit",
    "B_bit",
    "score_NBR",
    "score_NDWI",
    "score_NDMI",
    "score_NDVI",
    "score_clay",
    "score_amplitude",
]

TARGET = "density"  # <-- set your real column name here

X = df[features]
y = df[TARGET].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=14,
    random_state=42
)
model.fit(X_train, y_train)

joblib.dump(model, "density_regressor.pkl")

# Predictions
y_pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True density")
plt.ylabel("Predicted density")
plt.title("Prediction vs True")
plt.show()
