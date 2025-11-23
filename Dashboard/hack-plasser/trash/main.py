import math
import sys

import pandas as pd
import pulp

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/data1.csv"

# ========== Parameters (tune these) ==========
BUDGET = 200000           # total budget available for maintenance (currency units)
MAX_JOBS = 500            # maximum number of maintenance actions we can schedule
GROUP_DISCOUNT = 0.25     # fractional discount on maintenance cost if we perform any job in a group
GROUP_KM_WINDOW = 0.01    # kilometers — grouping window (e.g. 0.01 km = 10 m)
BASE_MAINT_COST = 400     # base cost per maintenance action (currency units)
W_TAMP_WEAR = 2.0         # weight converting tamp_tool_wear_pct into cost units
W_CYCLES = 0.5            # weight for cycles_per_sleeper
W_PRIORITY = 10.0         # weight for maintenance_priority_score
ALPHA_SETTLEMENT = 8.0    # penalty per mm predicted 30-day settlement if not repaired
VOID_PENALTY = 300.0      # penalty if void_detected and not repaired
UT_DEFECT_PENALTY = 400.0 # penalty if ut_defect_flag and not repaired
MIN_DAYS_SINCE_MAINT = 7  # don't do maintenance if days_since_maintenance is less than this (optional constraint)
# =============================================

# Load CSV
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"], dayfirst=False)
df.index = range(len(df))
n = len(df)
if n == 0:
    raise SystemExit("CSV empty or not loaded")

# Normalize / cast flags to 0/1
def to_flag(col):
    return df[col].fillna(0).astype(int).clip(0,1)

# Some columns may be missing; create defaults if absent
for col in ("tamp_tool_wear_pct","cycles_per_sleeper","maintenance_priority_score",
            "pred_settlement_30d_mm","void_detected","ut_defect_flag","days_since_maintenance"):
    if col not in df.columns:
        df[col] = 0

df["void_detected"] = to_flag("void_detected")
df["ut_defect_flag"] = to_flag("ut_defect_flag")
df["days_since_maintenance"] = df["days_since_maintenance"].fillna(9999)

# Build per-record maintenance cost estimate and penalty if NOT maintained
def compute_cost(row):
    cost = BASE_MAINT_COST
    cost += W_TAMP_WEAR * float(row.get("tamp_tool_wear_pct", 0))
    cost += W_CYCLES * float(row.get("cycles_per_sleeper", 0))
    cost += W_PRIORITY * float(row.get("maintenance_priority_score", 0))
    return max(0.0, float(cost))

def compute_penalty(row):
    p = ALPHA_SETTLEMENT * float(row.get("pred_settlement_30d_mm", 0))
    p += VOID_PENALTY * int(row.get("void_detected", 0))
    p += UT_DEFECT_PENALTY * int(row.get("ut_defect_flag", 0))
    # optionally increase penalty if sleeper is old or travel speed high etc.
    if "sleeper_age_years" in row and not pd.isna(row["sleeper_age_years"]):
        p += 0.5 * float(row["sleeper_age_years"])
    return max(0.0, float(p))

df["_maint_cost"] = df.apply(compute_cost, axis=1)
df["_penalty_no_maint"] = df.apply(compute_penalty, axis=1)

# Create groups by rounding km_post to nearest window.
# If km_post missing, group by sleeper_id or job_id fallback
if "km_post" not in df.columns:
    df["km_post"] = 0.0

def km_group(km):
    try:
        return round(float(km) / GROUP_KM_WINDOW) * GROUP_KM_WINDOW
    except Exception:
        return 0.0

df["_group_key"] = df["km_post"].apply(km_group).astype(str) + "_" + df["job_id"].fillna("nj").astype(str)

groups = df["_group_key"].unique().tolist()
group_to_indices = {g: df.index[df["_group_key"] == g].tolist() for g in groups}

# Build MIP
model = pulp.LpProblem("maintenance_scheduling", pulp.LpMinimize)

# Decision vars: x_i = 1 if we maintain record i
x = pulp.LpVariable.dicts("x", (i for i in df.index), lowBound=0, upBound=1, cat="Binary")

# Group binary vars y_g = 1 if ANY record in group g is maintained
y = pulp.LpVariable.dicts("y", (g for g in groups), lowBound=0, upBound=1, cat="Binary")

# Objective: minimize cost_of_maint + penalty_for_not_maintained - group_discount*y_g
# cost_of_maint = sum(maint_cost_i * x_i)
# penalty_for_not_maintained = sum(penalty_i * (1 - x_i)) = sum(penalty_i) - sum(penalty_i * x_i)
total_maint_cost = pulp.lpSum(df.loc[i, "_maint_cost"] * x[i] for i in df.index)
total_penalty_if_skipped = pulp.lpSum(df.loc[i, "_penalty_no_maint"] * (1 - x[i]) for i in df.index)
# group discounts reduce objective when we perform any job in that group
discount_terms = pulp.lpSum(GROUP_DISCOUNT * sum(df.loc[i, "_maint_cost"] for i in group_to_indices[g]) * y[g]
                            for g in groups)

model += total_maint_cost + total_penalty_if_skipped - discount_terms, "total_expected_loss_and_cost"

# Constraints:
# 1) Budget constraint: total maintenance spending must be <= BUDGET
model += total_maint_cost <= BUDGET, "budget_limit"

# 2) Max number of jobs
model += pulp.lpSum(x[i] for i in df.index) <= MAX_JOBS, "max_jobs"

# 3) Link x_i and group var y_g: if any x_i in group -> y_g = 1; and if y_g = 1 then at least one x_i >= y_g
for g, indices in group_to_indices.items():
    # each x_i <= y_g (so y_g must be 1 when any x_i = 1)
    for i in indices:
        model += x[i] <= y[g], f"link_x{i}_to_y_{g}"
    # y_g <= sum(x_i)  (if group is selected, at least one x must be 1)
    model += pulp.lpSum(x[i] for i in indices) >= y[g], f"group_active_requires_some_x_{g}"

# 4) Avoid doing maintenance on records with very recent maintenance
for i in df.index:
    if df.loc[i, "days_since_maintenance"] < MIN_DAYS_SINCE_MAINT:
        model += x[i] == 0, f"recently_serviced_no_action_{i}"

# 5) (Optional) If we want to ensure certain high-priority items are forced to be considered:
# Force maintain if penalty massively exceeds maint cost (safety override)
OVERRIDE_RATIO = 5.0
for i in df.index:
    if df.loc[i, "_penalty_no_maint"] > OVERRIDE_RATIO * df.loc[i, "_maint_cost"]:
        model += x[i] == 1, f"forced_maint_{i}"

# Solve
solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=300)
result = model.solve(solver)

print("Status:", pulp.LpStatus[model.status])
print("Objective (total expected cost):", pulp.value(model.objective))
selected = [i for i in df.index if pulp.value(x[i]) > 0.5]

print(f"Selected {len(selected)} records for maintenance (budget {BUDGET}):")
out_df = df.loc[selected].copy()
out_df["_selected"] = 1
# show key columns and estimated cost/penalty
cols_to_show = ["timestamp","machine_id","km_post","sleeper_id","_maint_cost","_penalty_no_maint",
                "tamp_tool_wear_pct","cycles_per_sleeper","maintenance_priority_score","void_detected","ut_defect_flag","days_since_maintenance"]
cols_to_show = [c for c in cols_to_show if c in out_df.columns]
print(out_df[cols_to_show].head(200).to_string(index=False))

# Summary totals
total_spend = sum(df.loc[i, "_maint_cost"] for i in selected)
total_saved_penalty = sum(df.loc[i, "_penalty_no_maint"] for i in selected)
print(f"Estimated maintenance spend: {total_spend:.2f}")
print(f"Estimated penalties avoided (if we maintain those): {total_saved_penalty:.2f}")

# Save decisions
df["_selected"] = 0
df.loc[selected, "_selected"] = 1
df.to_csv("maint_decisions.csv", index=False)
print("Saved decisions to maint_decisions.csv")
