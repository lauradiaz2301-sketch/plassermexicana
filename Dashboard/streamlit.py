import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

def find_any(pathname):
    """
    Busca pathname en rutas probables y, si no aparece, hace una búsqueda recursiva
    dentro de la carpeta hack-plasser y en el cwd.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent  # /Users/.../Hackaton/Dashboard
    hack_plasser = repo_root / "hack-plasser"

    candidates = [
        here / pathname,
        repo_root / pathname,
        hack_plasser / "sat" / pathname,
        hack_plasser / "data" / pathname,
        here / "hack-plasser" / "sat" / pathname,
        here / "hack-plasser" / "data" / pathname,
        Path("/Users/schns/Documents/Hackaton/Dashboard/hack-plasser/sat") / pathname,
        Path("/Users/schns/Documents/Hackaton/Dashboard/hack-plasser/data") / pathname,
        Path.home() / "Downloads" / "hack-plasser" / "sat" / pathname,
        Path(pathname),
    ]
    for p in candidates:
        try:
            if p.exists():
                return str(p.resolve())
        except Exception:
            continue

    # búsqueda recursiva dentro de hack-plasser si existe
    if hack_plasser.exists():
        for p in hack_plasser.rglob(pathname):
            return str(p.resolve())

    # búsqueda recursiva en cwd como último recurso
    for p in Path.cwd().rglob(pathname):
        return str(p.resolve())

    return None

# --- bloque: cargar modelo + datos y dibujar scatter True vs Predicted ---
model_p = find_any("density_regressor.pkl")
data_p = find_any("synthetic_timeseries_with_priority_info.csv")

if model_p and data_p:
    try:
        model = joblib.load(model_p)
        df_model = pd.read_csv(data_p)
        features = [
            "NDVI","NDWI","NDMI","NBR","clay_frac","mm_interval","mm_cumulative",
            "VV_bit","VH_bit","B_bit","score_NBR","score_NDWI","score_NDMI",
            "score_NDVI","score_clay","score_amplitude",
        ]
        if not set(features).issubset(df_model.columns) or "density" not in df_model.columns:
            st.info("Columnas requeridas faltan en el CSV del modelo; no se puede trazar la gráfica.")
    except Exception as e:
        st.error(f"No se pudo generar la gráfica del modelo: {e}")
else:
    missing = []
    if not model_p:
        missing.append("density_regressor.pkl")
    if not data_p:
        missing.append("synthetic_timeseries_with_priority_info.csv")
    st.info(f"Modelo o datos no encontrados: faltan {', '.join(missing)}. Busqué en hack-plasser/sat, hack-plasser/data y rutas comunes.")

st.set_page_config(layout="wide", page_title="Railway Maintenance Dashboard")

# -----------------------------------------------------------
# 1) LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    sensors = pd.read_csv("Bad Track New Coordinates.csv")
    rail = pd.read_csv("via_ferrea_points (1).csv")
    return sensors, rail

df, rail = load_data()

# -----------------------------------------------------------
# 2) PREPARE RAIL GEOMETRY (robusto)
# -----------------------------------------------------------

# Drop invalid values
rail = rail.dropna(subset=["longitude", "latitude"])

# Ensure numeric types
rail["longitude"] = pd.to_numeric(rail["longitude"], errors="coerce")
rail["latitude"] = pd.to_numeric(rail["latitude"], errors="coerce")

# convert 'order' only if the column actually exists
if "order" in rail.columns:
    rail["order"] = pd.to_numeric(rail["order"], errors="coerce")

# Sidebar selection of line
st.sidebar.title("Filters")
selected_line = st.sidebar.selectbox("Select Rail Line", rail["line_id"].unique())

# Subset for this line (keep rows with coords)
line_subset = rail[rail["line_id"] == selected_line].dropna(subset=["longitude", "latitude"]).copy()

# Detect if lon/lat are possibly swapped (common issue)
# If longitudes look like latitudes (all in [-90,90]) and latitudes are outside that, swap columns.
lon_abs_max = line_subset["longitude"].abs().max() if len(line_subset) > 0 else 0
lat_abs_max = line_subset["latitude"].abs().max() if len(line_subset) > 0 else 0
if lon_abs_max <= 90 and lat_abs_max > 90:
    line_subset = line_subset.rename(columns={"longitude": "latitude_tmp", "latitude": "longitude"})
    line_subset = line_subset.rename(columns={"latitude_tmp": "latitude"})

# Build ordered dataframe using 'order' if meaningful, otherwise try km_post or index
use_order = ("order" in line_subset.columns) and line_subset["order"].notna().any()
if use_order:
    line_df = line_subset.sort_values("order").copy()
else:
    if "km_post" in line_subset.columns and line_subset["km_post"].notna().any():
        line_df = line_subset.sort_values("km_post").copy()
    else:
        line_df = line_subset.sort_index().copy()

# Remove duplicates carefully: by order if used, otherwise by coords
if use_order:
    line_df = line_df.drop_duplicates(subset=["order"])
else:
    line_df = line_df.drop_duplicates(subset=["longitude", "latitude"])

# Build ordered path: list of [lon, lat] floats
path_coords = []
for _, r in line_df.loc[:, ["longitude", "latitude"]].iterrows():
    try:
        lon = float(r["longitude"])
        lat = float(r["latitude"])
        path_coords.append([lon, lat])
    except Exception:
        continue

# If path is extremely long, downsample while preserving order
max_points = 2000
if len(path_coords) > max_points:
    idx = np.linspace(0, len(path_coords) - 1, max_points, dtype=int)
    path_coords = [path_coords[i] for i in idx]

# Create PathLayer only if we have at least two valid points
if len(path_coords) >= 2:
    line_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path_coords}],
        get_path="path",
        get_width=6,
        width_units="pixels",
        get_color=[10, 130, 200],  # bright blue
        rounded=True,
        capRounded=True,
        pickable=False
    )
else:
    line_layer = None

# -----------------------------------------------------------
# 3) FILTER SENSOR DATA
# -----------------------------------------------------------

machine = st.sidebar.multiselect("Machine ID", df["machine_id"].unique())
mode = st.sidebar.multiselect("Tamping Mode", df["tamp_mode"].unique())
ut_flag = st.sidebar.selectbox("UT Defect", ["All", 0, 1])
wim_flag = st.sidebar.selectbox("WIM Detection", ["All", 0, 1])

filtered = df.copy()

if machine:
    filtered = filtered[filtered["machine_id"].isin(machine)]
if mode:
    filtered = filtered[filtered["tamp_mode"].isin(mode)]
if ut_flag != "All":
    filtered = filtered[filtered["ut_defect_flag"] == ut_flag]
if wim_flag != "All":
    filtered = filtered[filtered["wimdetection_flag"] == wim_flag]

# Avoid massive sends: keep only some points
filtered_small = filtered.iloc[::20].copy()

# Priority normalization for heatmap
filtered_small["priority_norm"] = (
    (filtered_small["maintenance_priority_score"] - filtered_small["maintenance_priority_score"].min()) /
    (filtered_small["maintenance_priority_score"].max() - filtered_small["maintenance_priority_score"].min())
).fillna(0)

# -----------------------------------------------------------
# 4) KPIs
# -----------------------------------------------------------
st.title("Railway Maintenance Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rigidez Balasto (kN/mm)", f"{filtered_small['balast_rigidity_kN_per_mm'].mean():.2f}")
col2.metric("Fouling Index (%)", f"{filtered_small['fouling_index_pct'].mean():.1f}")
col3.metric("Energy Compaction (J)", f"{filtered_small['energy_compaction_J'].mean():.0f}")
col4.metric("Temp Hidráulica (°C)", f"{filtered_small['hydraulic_oil_temp_C'].mean():.1f}")

# -----------------------------------------------------------
# 5) MAP LAYERS (render)
# -----------------------------------------------------------

points_layer = pdk.Layer(
    "ScatterplotLayer",
    filtered_small,
    get_position=["lon", "lat"],
    get_radius=250,
    radius_min_pixels=4,
    radius_max_pixels=100,
    get_color="[255 * priority_norm, 50, 255 * (1 - priority_norm)]",
    pickable=True,
)

# Map view: compute center from combined data (path + filtered points) and pick zoom to fit
sensor_coords = []
if len(filtered_small) > 0 and {"lon", "lat"}.issubset(filtered_small.columns):
    sensor_coords = filtered_small.loc[:, ["lon", "lat"]].dropna().apply(lambda r: [float(r["lon"]), float(r["lat"])], axis=1).tolist()

combined_coords = []
if len(path_coords) > 0:
    combined_coords.extend(path_coords)
combined_coords.extend(sensor_coords)

if len(combined_coords) > 0:
    lons = [p[0] for p in combined_coords]
    lats = [p[1] for p in combined_coords]
    center_lon = float(np.mean(lons))
    center_lat = float(np.mean(lats))
    lon_span = max(lons) - min(lons)
    lat_span = max(lats) - min(lats)
    span = max(lon_span, lat_span)

    # heuristic zoom by span (degrees)
    if span <= 0.005:
        zoom = 15
    elif span <= 0.02:
        zoom = 14
    elif span <= 0.05:
        zoom = 13
    elif span <= 0.2:
        zoom = 12
    elif span <= 0.5:
        zoom = 10
    elif span <= 1.5:
        zoom = 8
    elif span <= 5:
        zoom = 6
    else:
        zoom = 4
else:
    # fallback to rail centroid / default zoom
    center_lat = float(rail["latitude"].mean())
    center_lon = float(rail["longitude"].mean())
    zoom = 7

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom,
    pitch=45
)

# Ensure we have a PathLayer built from path_coords and include it in layers
# (this patch guarantees the rail line is added to the map)
if "path_coords" in globals() and isinstance(path_coords, list) and len(path_coords) >= 2:
    line_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path_coords}],
        get_path="path",
        get_width=6,
        width_units="pixels",
        get_color=[10, 130, 200],  # visible blue
        rounded=True,
        capRounded=True,
        pickable=False
    )
else:
    line_layer = None

# Build layers: put the rail line first (below points)
layers = []
if line_layer is not None:
    layers.append(line_layer)
if 'points_layer' in globals() and points_layer is not None:
    layers.append(points_layer)

st.subheader("Mapa de Vía + Sensores")
st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "Sleeper: {sleeper_id}\nPriority: {maintenance_priority_score}\nKM: {km_post}"}
    )
)

# -----------------------------------------------------------
# DATOS FILTRADOS
# -----------------------------------------------------------
st.subheader("Datos Filtrados")
st.dataframe(filtered)

# -----------------------------------------------------------
# RIGIDEZ vs COMPACTION (scatter)
# -----------------------------------------------------------
if len(filtered_small) > 0 and {"balast_rigidity_kN_per_mm", "energy_compaction_J"}.issubset(filtered_small.columns):
    st.subheader("Rigidez vs Compaction Energy")
    chart_rig = (
        alt.Chart(filtered_small)
        .mark_circle(size=100)
        .encode(
            x=alt.X("balast_rigidity_kN_per_mm:Q", title="Rigidez (kN/mm)"),
            y=alt.Y("energy_compaction_J:Q", title="Energy (J)"),
            color=alt.Color("maintenance_priority_score:Q", title="Priority"),
            tooltip=["machine_id", "km_post", "sleeper_id"]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_rig, use_container_width=True)
else:
    st.info("No hay datos suficientes para Rigidez vs Compaction (columnas faltantes).")

# -----------------------------------------------------------
# TENDENCIAS Y RELACIONES
# -----------------------------------------------------------
st.subheader("Tendencias y Relaciones")
if len(filtered_small) > 0:
    chart_trend = (
        alt.Chart(filtered_small)
        .mark_line()
        .encode(
            x="timestamp:T",
            y="pred_settlement_30d_mm:Q",
            color="machine_id:N"
        )
        .properties(height=250, title="Predicción Asentamiento 30 días")
    )

    # NOTE: removed duplicate Rigidez vs Compaction chart (kept in its own section)
    st.altair_chart(chart_trend, use_container_width=True)
else:
    st.info("No hay datos filtrados para graficar tendencias.")

# -----------------------------------------------------------
# MODELO: TRUE vs PREDICTED (density) - requiere model + CSV
# -----------------------------------------------------------
# (usa la función find_any y el bloque joblib + pyplot ya definidos arriba)
model_p = find_any("density_regressor.pkl")
data_p = find_any("synthetic_timeseries_with_priority_info.csv")

if model_p and data_p:
    try:
        model = joblib.load(model_p)
        df_model = pd.read_csv(data_p)
        features = [
            "NDVI","NDWI","NDMI","NBR","clay_frac","mm_interval","mm_cumulative",
            "VV_bit","VH_bit","B_bit","score_NBR","score_NDWI","score_NDMI",
            "score_NDVI","score_clay","score_amplitude",
        ]
        if not set(features).issubset(df_model.columns) or "density" not in df_model.columns:
            st.info("Datos incompletos en CSV del modelo; no se puede dibujar True vs Predicted.")
        else:
            Xm = df_model[features].astype(float).fillna(0)
            y_true = df_model["density"].astype(float).fillna(0)
            y_pred = model.predict(Xm)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#2b8cbe")
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="#f03b20", lw=1.5)
            ax.set_xlabel("True density")
            ax.set_ylabel("Predicted density")
            ax.set_title("True vs Predicted: density")
            plt.tight_layout()
            st.subheader("Modelo: True vs Predicted (density)")
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.error(f"No se pudo generar la gráfica del modelo: {e}")
else:
    st.info("Modelo o datos para True vs Predicted no encontrados. Busqué density_regressor.pkl y synthetic_timeseries_with_priority_info.csv.")

# -----------------------------------------------------------
# PREDICTED DENSITY: DISTRIBUCIÓN GAUSSIANA + TOP10%
# -----------------------------------------------------------
pred_csv = find_any("sat_predictions_output.csv")
pred_img = find_any("sat_pred_density_distribution.png")

if pred_csv:
    try:
        df_pred = pd.read_csv(pred_csv)
        pred_col = next((c for c in ["pred_density", "predicted_density", "predicted"] if c in df_pred.columns), None)
        if pred_col is None:
            st.info("CSV de predicciones encontrado pero no contiene 'pred_density'.")
        else:
            vals = df_pred[pred_col].dropna().astype(float)
            if vals.empty:
                st.info("La columna de predicción está vacía.")
            else:
                mu = vals.mean()
                sigma = vals.std(ddof=0)
                fig, ax = plt.subplots(figsize=(8, 3.5))
                ax.hist(vals, bins=30, density=True, alpha=0.6, color="#4C72B0", edgecolor="white")
                x = np.linspace(vals.min(), vals.max(), 200)
                if sigma > 0:
                    pdf = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    ax.plot(x, pdf, color="#DD8452", lw=2, label=f"N(μ={mu:.3f}, σ={sigma:.3f})")
                    ax.legend()
                ax.set_title("Distribución de Predicted density (hist + ajuste normal)")
                ax.set_xlabel(pred_col)
                ax.set_ylabel("Density")
                plt.tight_layout()
                st.subheader("Predicted density: distribución Gaussiana")
                st.pyplot(fig)
                plt.close(fig)
    except Exception as e:
        st.error(f"No se pudo leer/plotear {pred_csv}: {e}")

elif pred_img:
    st.subheader("Predicted density: distribución (imagen)")
    st.image(pred_img, use_container_width=True)

    # intentar cargar CSV junto a la imagen para la tabla top10
    img_path = Path(pred_img)
    maybe_csv = img_path.with_name("sat_predictions_output.csv")
    if maybe_csv.exists():
        try:
            df_pred = pd.read_csv(str(maybe_csv))
            pred_col = next((c for c in ["pred_density", "predicted_density", "predicted"] if c in df_pred.columns), None)
            if pred_col:
                q90 = float(df_pred[pred_col].quantile(0.9))
                top10 = df_pred[df_pred[pred_col] >= q90].sort_values(pred_col, ascending=False)
                st.subheader("Top 10% (desde CSV junto a la imagen)")
                st.write(f"Threshold (0.9 quantile): {q90:.6f} — Filas: {len(top10)}")
                st.dataframe(top10)
                csv_bytes = top10.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV (top10%)", data=csv_bytes, file_name="sat_top10_percent_predictions.csv", mime="text/csv")
        except Exception:
            pass
else:
    st.info("No se encontró sat_predictions_output.csv ni sat_pred_density_distribution.png. Ejecuta hack-plasser/sat/test_mode.py para generarlos.")

# -----------------------------------------------------------
# SEGMENTS NEEDING MAINTENANCE (si existe segments CSV)
# -----------------------------------------------------------
segments_csv = find_any("sat_segments_for_maintenance.csv")
if segments_csv:
    try:
        df_seg = pd.read_csv(segments_csv)
        st.subheader("Segments needing maintenance (top 10%)")
        st.write(f"Filas: {len(df_seg)}")
        st.dataframe(df_seg)
        csv_bytes = df_seg.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV (segments_for_maintenance)", data=csv_bytes, file_name="sat_segments_for_maintenance.csv", mime="text/csv")
    except Exception as e:
        st.error(f"No se pudo leer {segments_csv}: {e}")
else:
    # si no existe, intentar usar top10 ya calculado arriba (pred_csv case)
    if 'top10' in locals() and not top10.empty:
        st.subheader("Segments needing maintenance (derivado de top10%)")
        st.dataframe(top10)
        csv_bytes = top10.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar CSV (segments_derived_from_top10)", data=csv_bytes, file_name="sat_segments_from_top10.csv", mime="text/csv")
    else:
        st.info("No se encontró archivo de segmentos de mantenimiento ni top10 disponible.")
