import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)

MODEL_PATH = "railroad_classifier.pkl"
TEST_CSV = "../data/DatasetErroresSinClasificar.csv"

features = [
    "pre_level_mm","post_level_mm","pre_align_mm","post_align_mm",
    "gauge_mm","cant_mm",
    "squeeze_pressure_bar","squeeze_angle_deg",
    "lift_force_kN","align_force_kN","cycle_duration_ms"
]

# ----------------------------
# Load data + model
# ----------------------------
model = joblib.load(MODEL_PATH)
df = pd.read_csv(TEST_CSV)

X_test = df[features]

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Save predictions
df["pred"] = y_pred
df["prob_bad"] = y_prob
df.to_csv("classified_output.csv", index=False)
print("classified_output.csv written!")

# ----------------------------
# Summary statistics
# ----------------------------
print("\n===== Prediction Distribution =====")
print(df["pred"].value_counts(normalize=True).rename(lambda x: f"class {x}"))

print("\n===== Predicted Probability Summary =====")
print(df["prob_bad"].describe())

# ----------------------------
# Confusion matrix
# (only if your test CSV already contains real labels)
# ----------------------------
if "label" in df.columns:
    print("\n===== Classification Report =====")
    print(classification_report(df["label"], y_pred))

    cm = confusion_matrix(df["label"], y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(df["label"], y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Good/bad datasets check
# ----------------------------
def check_mean(path):
    g = pd.read_csv(path)
    g = g[features]
    return model.predict(g).mean()

print("\nmean(pred) good.csv =", check_mean("../data/good.csv"))
print("mean(pred) bad.csv  =", check_mean("../data/bad.csv"))
