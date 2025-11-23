import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# 1. Load CSV
# ------------------------------------------------------
df = pd.read_csv("data/data1_100.csv")

# ------------------------------------------------------
# 2. Select HMM Features
#    DO NOT include timestamps, ids, strings
# ------------------------------------------------------
features = [
    "pre_level_mm",
    "post_level_mm",
    "pre_align_mm",
    "post_align_mm",
    "gauge_mm",
    "cant_mm",
    "squeeze_pressure_bar",
    "lift_force_kN",
    "align_force_kN",
    "cycle_duration_ms",
    "vibration_peak_g",
    "contactless_duration_ms",
    "energy_compaction_J",
    "void_detected",
    "cycles_per_sleeper",
    "ballast_density_kgm3",
    "ballast_moisture_pct",
    "fouling_index_pct",
    "balast_rigidity_kN_per_mm",
    "tamp_tool_wear_pct",
    "travel_speed_kmh",
    "stabilizer_amplitude_mm",
]

X = df[features].replace({True: 1, False: 0}).fillna(method="ffill").values

# ------------------------------------------------------
# 3. Standardize Inputs
# ------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------
# 4. Fit the HMM
#    n_components = number of latent operational modes
# ------------------------------------------------------
model = GaussianHMM(
    n_components=4,          # adjust depending on your machines
    covariance_type="full",
    n_iter=200,
    random_state=42
)

model.fit(X_scaled)

# ------------------------------------------------------
# 5. Infer hidden state for each row
# ------------------------------------------------------
hidden_states = model.predict(X_scaled)
df["hmm_state"] = hidden_states

print(df[["timestamp", "machine_id", "hmm_state"]].head())

# ------------------------------------------------------
# 6. Example: Probability of each state
# ------------------------------------------------------
#print("State means:")
#print(model.means_)
#print(len( model.means_ ))
#print(len( model.means_[0] ))


# ------------------------------------------------------
# prediction
# ------------------------------------------------------
current_state = hidden_states[-1]

# Forecast t steps ahead
t = 2000  # number of future timesteps (depends on sampling rate)

# Compute transition matrix power
Tt = np.linalg.matrix_power(model.transmat_, t)

# Distribution of states after t steps
future_dist = Tt[current_state]

# Expected future feature vector (in scaled space)
future_scaled = np.zeros(model.means_.shape[1])
for i in range(model.n_components):
    future_scaled += future_dist[i] * model.means_[i]

# Convert back to real values
future_real = scaler.inverse_transform(future_scaled.reshape(1, -1))

print("Predicted feature values in t steps:")
print(future_real)
