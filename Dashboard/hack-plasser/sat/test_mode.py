import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load model and data
MODEL_PATH = "sat_classifier.pkl"  # or your regressor .pkl
CSV_PATH = "../data/sat.csv"

features = [
    "NDVI","NDWI","NDMI","NBR","clay_frac",
    "mm_interval","mm_cumulative",
    "VV_bit","VH_bit","B_bit",
    "score_NBR","score_NDWI","score_NDMI",
    "score_NDVI","score_clay","score_amplitude"
]

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=features)

# Load regressor
model = joblib.load(MODEL_PATH)

# Select pseudo-unseen subset
test = df.sample(frac=0.2, random_state=42)
X_test = test[features]

# Predict continuous target
test["pred_density"] = model.predict(X_test)

# Export results
test.to_csv("sat_predictions_output.csv", index=False)
print("Saved predictions to sat_predictions_output.csv")

# Plot predicted density distribution
plt.figure(figsize=(10,5))
sns.histplot(test["pred_density"], bins=30, kde=True)
plt.title("Predicted Density Distribution")
plt.xlabel("Predicted density")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sat_pred_density_distribution.png")
#plt.show()


# -- select maintenance
# Load predictions
df = pd.read_csv("./sat_predictions_output.csv")  # path to your previous predictions

threshold = df["pred_density"].quantile(0.9)
MAINTENANCE_THRESHOLD = threshold

print(df.columns.tolist())
df_to_maintain = df[df["pred_density"] >= MAINTENANCE_THRESHOLD]
print('check ',df["pred_density"].describe())


# Select identifying columns + predicted density
columns_to_export = ["id", "timestamp", "site_id", "synth_point_id", "lat", "lon", "pred_density"]
df_to_maintain = df_to_maintain[columns_to_export]

# Export
df_to_maintain.to_csv("sat_segments_for_maintenance.csv", index=False)
print("Exported segments needing maintenance: sat_segments_for_maintenance.csv")

# Optional: view top 10
print(df_to_maintain.sort_values("pred_density", ascending=False).head(10))
