import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load raw data
good = pd.read_csv('../data/good.csv')
bad  = pd.read_csv('../data/bad.csv')

# add noise
noise_pct = 0.05
def add_noise(df, feature, pct=noise_pct):
    #sigma = df[feature].std() * pct
    gap = abs(good[f].mean() - bad[f].mean())
    sigma = gap * noise_factor   # e.g., 0.5 → overlap
    df[feature] += np.random.normal(0, sigma, len(df))
# EXACT features for model
features = [
    "pre_level_mm","post_level_mm","pre_align_mm","post_align_mm",
    "gauge_mm","cant_mm",
    "squeeze_pressure_bar","squeeze_angle_deg",
    "lift_force_kN","align_force_kN","cycle_duration_ms"
]
n = 3000  # rows per class
good = pd.DataFrame({
    "pre_level_mm": np.random.normal(0, 4, size=n),
    "post_level_mm": np.random.normal(0, 4, size=n),
    "pre_align_mm": np.random.normal(0, 3, size=n),
    "post_align_mm": np.random.normal(0, 3, size=n),
    "gauge_mm": np.random.normal(1435, 5, size=n),
    "cant_mm": np.random.normal(150, 10, size=n),
    "squeeze_pressure_bar": np.random.normal(130, 10, size=n),
    "squeeze_angle_deg": np.random.normal(40, 5, size=n),
    "lift_force_kN": np.random.normal(30, 5, size=n),
    "align_force_kN": np.random.normal(15, 4, size=n),
    "cycle_duration_ms": np.random.normal(900, 80, size=n)
})
n = 1500  # rows per class
bad = pd.DataFrame({
    "pre_level_mm": np.random.normal(0, 4, size=n),  # overlaps with good!
    "post_level_mm": np.random.normal(0, 4, size=n),
    "pre_align_mm": np.random.normal(0, 3, size=n),
    "post_align_mm": np.random.normal(0, 3, size=n),
    "gauge_mm": np.random.normal(1435, 6, size=n),
    "cant_mm": np.random.normal(150, 12, size=n),
    "squeeze_pressure_bar": np.random.normal(140, 12, size=n),  # slightly higher mean
    "squeeze_angle_deg": np.random.normal(42, 6, size=n),
    "lift_force_kN": np.random.normal(32, 6, size=n),
    "align_force_kN": np.random.normal(18, 5, size=n),
    "cycle_duration_ms": np.random.normal(970, 100, size=n)
})

good["label"] = 0
bad["label"]  = 1

df = pd.concat([good, bad], ignore_index=True)


# ✨ FORCE dataframe to contain ONLY these columns
df = df[features + ["label"]]

# check for leakage
corr = df.corr()
print('check leakage::')
print(corr["label"].sort_values(ascending=False))

# plot features
num_features = len(features)
cols = 3  # how many plots per row
rows = (num_features + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
axes = axes.flatten()
for i, f in enumerate(features):
    sns.boxplot(data=df, x='label', y=f, ax=axes[i])
    axes[i].set_title(f)
plt.tight_layout()
plt.suptitle("")
plt.savefig('features.png')

# Training data
X = df[features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "railroad_classifier.pkl")
print("MODEL SAVED!")


# --- Evaluate ---
y_pred = model.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))

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
plt.savefig('confusion-matrix.png')
#plt.show()

