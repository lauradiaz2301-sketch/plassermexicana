import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Load data ---
good = pd.read_csv('data/good.csv')
bad = pd.read_csv('data/bad.csv')

# Add labels
good["label"] = 0
bad["label"] = 1

df = pd.concat([good, bad], ignore_index=True)

# --- Select numeric features (drop strings, IDs, timestamps, categories) ---

# Remove columns that are clearly non-features
drop_cols = [
    "timestamp", "machine_id", "sleeper_id", "operator_id", "job_id",
    "firmware_version", "travel_direction", "rail_defect_type",
    "sleeper_material", "sleeper_condition"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categorical columns that remain (if any)
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features / target
X = df.drop(columns=["label"])
y = df["label"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- Train classifier ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Save model ---
joblib.dump(model, "railroad_classifier.pkl")

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks([0,1])
ax.set_yticks([0,1])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")
plt.title("Confusion Matrix")
plt.show()

# --- Export predictions to CSV ---
#preds_df = X_test.copy()
#preds_df["true_label"] = y_test
#preds_df["predicted_label"] = y_pred
#preds_df.to_csv("predictions.csv", index=False)
#
#print("Model saved to railroad_classifier.pkl")
#print("Predictions saved to predictions.csv")
