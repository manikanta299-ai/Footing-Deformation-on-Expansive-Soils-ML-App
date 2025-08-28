# app.py  —  Footing deformation predictor (SVR / RandomForest)
# ------------------------------------------------------------------
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple

st.set_page_config(
    page_title="Footing Deformation on Expansive Soils — ML App",
    layout="wide",
)

# --------- Canonical training feature order (IMPORTANT) ----------
FEATURE_NAMES: List[str] = [
    "Day",
    "Temperature_˚C",   # note the degree symbol
    "Rainfall_mm",
    "SWC_Footing_1_%",
    "SWC_Footing_2_%",
    "SWC_Footing_3_%",
    "SWC_Footing_4_%",
]

NON_ENSEMBLE_NEED_SCALING = {"SVR"}   # only SVR uses the scaler

# Expected test R² for display (edit to your exact values if you like)
EXPECTED_R2: Dict[str, List[float]] = {
    "SVR":          [0.96, 0.95, 0.97, 0.96],
    "RandomForest": [0.97, 0.95, 0.96, 0.96],
}

# -------------------- model loading ------------------------------
@st.cache_resource
def load_artifacts() -> Tuple[Dict[str, object], object]:
    models = {}
    # Prefer joblib artifacts you converted (compatible across sklearn versions)
    try:
        models["SVR"] = joblib.load("SVR.joblib")
    except Exception as e:
        st.warning(f"Could not load SVR.joblib ({e}).")
    try:
        models["RandomForest"] = joblib.load("RandomForest.joblib")
    except Exception as e:
        st.warning(f"Could not load RandomForest.joblib ({e}).")
    scaler = None
    try:
        scaler = joblib.load("scaler.joblib")
    except Exception as e:
        st.warning(f"Could not load scaler.joblib ({e}). SVR requires it.")
    return models, scaler

MODELS, SCALER = load_artifacts()

# --------------- utilities: header normalization -----------------
def _normalize_name(c: str) -> str:
    """make header comparisons robust (case, punctuation, degree symbol)."""
    x = c.strip()
    # unify degree symbols and remove duplicate underscores/spaces
    x = x.replace("°", "˚")        # normalize to U+02DA
    x = x.replace("degC", "˚C")
    x = x.replace(" ", "_")
    while "__" in x:
        x = x.replace("__", "_")
    return x

HEADER_ALIASES: Dict[str, List[str]] = {
    "Day": ["day", "Day"],
    "Temperature_˚C": [
        "Temperature_˚C", "Temperature_°C", "Temperature_C",
        "temperature_c", "Temp_˚C", "Temp_C"
    ],
    "Rainfall_mm": ["Rainfall_mm", "rain_mm", "Rain_mm", "Rainfall"],
    "SWC_Footing_1_%": ["SWC_Footing_1_%", "SWC_1", "SWC1_%"],
    "SWC_Footing_2_%": ["SWC_Footing_2_%", "SWC_2", "SWC2_%"],
    "SWC_Footing_3_%": ["SWC_Footing_3_%", "SWC_3", "SWC3_%"],
    "SWC_Footing_4_%": ["SWC_Footing_4_%", "SWC_4", "SWC4_%"],
}

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    norm_to_original = { _normalize_name(c): c for c in df.columns }
    for target, aliases in HEADER_ALIASES.items():
        found = None
        for a in aliases:
            key = _normalize_name(a)
            if key in norm_to_original:
                found = norm_to_original[key]
                break
        if found is not None:
            colmap[found] = target
    return df.rename(columns=colmap)

# -------------------- feature preparation ------------------------
def prepare_for_model(model_name: str, Xdf: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct order/dtype and scale only for SVR.
    Always return a DataFrame with FEATURE_NAMES as columns.
    """
    # Defensive: must be DataFrame
    if not isinstance(Xdf, pd.DataFrame):
        raise TypeError("prepare_for_model expects a pandas DataFrame.")

    df = Xdf.copy()
    # Exact training order
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")

    df = df[FEATURE_NAMES]
    # numeric
    for c in FEATURE_NAMES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise ValueError("NaNs found in input features after coercion.")

    # Scale for SVR only
    if model_name in NON_ENSEMBLE_NEED_SCALING:
        if SCALER is None:
            raise RuntimeError("Scaler not loaded but required for SVR.")
        try:
            # keep pandas output (sklearn >=1.2)
            SCALER.set_output(transform="pandas")
            out = SCALER.transform(df)
            out = out[FEATURE_NAMES]
        except Exception:
            # fallback for older versions
            out = pd.DataFrame(SCALER.transform(df.values), columns=FEATURE_NAMES)
        return out

    # RandomForest etc -> raw features
    return df

# --------------------- prediction helpers ------------------------
FOOT_COLUMNS = ["Footing_1_mm", "Footing_2_mm", "Footing_3_mm", "Footing_4_mm"]

def _predict(model_obj, X: pd.DataFrame) -> np.ndarray:
    """unified prediction: returns shape (n_samples, 4)."""
    y = model_obj.predict(X)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return y

def predict_single(model_key: str, values: Dict[str, float]) -> np.ndarray:
    """Build one-row DF -> prepare_for_model -> predict."""
    if model_key not in MODELS:
        raise RuntimeError(f"Model '{model_key}' is not loaded.")
    row = pd.DataFrame([values], columns=FEATURE_NAMES)
    Xready = prepare_for_model(model_key, row)   # << correct order of args
    return _predict(MODELS[model_key], Xready)   # (1,4)

# ---------------------------- UI ---------------------------------
st.title("Footing Deformation on Expansive Soils — ML App")
st.caption("Loads your saved SVR and RandomForest models (+ scaler for SVR) and predicts deformation (mm).")

left, right = st.columns([0.33, 0.67])

with left:
    model_choice = st.selectbox("Choose model", ["SVR", "RandomForest"])

    st.subheader("Feature Inputs")
    Day = st.number_input("Day", min_value=0, step=1, value=200)
    Temperature = st.number_input("Temperature (°C)", step=0.1, value=24.3)
    Rain = st.number_input("Rainfall (mm)", step=0.1, value=0.0)
    swc1 = st.number_input("SWC Footing 1 (%)", step=0.1, value=10.9)
    swc2 = st.number_input("SWC Footing 2 (%)", step=0.1, value=13.4)
    swc3 = st.number_input("SWC Footing 3 (%)", step=0.1, value=8.3)
    swc4 = st.number_input("SWC Footing 4 (%)", step=0.1, value=9.5)

with right:
    st.subheader("Single Prediction")
    try:
        vals = {
            "Day": Day,
            "Temperature_˚C": Temperature,
            "Rainfall_mm": Rain,
            "SWC_Footing_1_%": swc1,
            "SWC_Footing_2_%": swc2,
            "SWC_Footing_3_%": swc3,
            "SWC_Footing_4_%": swc4,
        }
        yhat = predict_single(model_choice, vals)[0]  # shape (4,)
        c1, c2, c3, c4 = st.columns(4)
        for i, (c, foot) in enumerate(zip([c1, c2, c3, c4], FOOT_COLUMNS)):
            with c:
                st.markdown(f"**{foot}**")
                st.markdown(f"<h2 style='margin:0'>{yhat[i]:.3f} mm</h2>", unsafe_allow_html=True)
                r2 = EXPECTED_R2.get(model_choice, [np.nan]*4)[i]
                st.caption(f"🟢 exp. R² ≈ {r2*100:.1f}%")
    except Exception as e:
        st.error(f"Prediction error: {e}")

    st.subheader("Batch Predictions and Diagnostics")
    with st.expander("Instructions / Template", expanded=False):
        st.markdown(
            """
            **Required columns** (any order; punctuation/case doesn’t matter):  
            `Day, Temperature_˚C, Rainfall_mm, SWC_Footing_1_%, SWC_Footing_2_%, SWC_Footing_3_%, SWC_Footing_4_%`

            **Optional ground-truth columns:** `Footing_1_mm, Footing_2_mm, Footing_3_mm, Footing_4_mm`
            """
        )
        # Downloadable template
        tmpl = pd.DataFrame(columns=FEATURE_NAMES + FOOT_COLUMNS)
        buff = io.BytesIO()
        tmpl.to_csv(buff, index=False)
        st.download_button("Download CSV Template", buff.getvalue(), file_name="batch_template.csv", mime="text/csv")

    up = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            df = canonicalize_columns(df)
            X = df.copy()
            # Keep only features (non-destructive: we’ll join predictions later)
            X = X[[c for c in FEATURE_NAMES if c in X.columns]]
            # Check missing
            missing = [c for c in FEATURE_NAMES if c not in X.columns]
            if missing:
                raise KeyError(f"Missing features: {missing}")

            Xready = prepare_for_model(model_choice, X)
            y_batch = _predict(MODELS[model_choice], Xready)   # (n,4)
            pred_df = pd.DataFrame(y_batch, columns=[f"pred_{c}" for c in FOOT_COLUMNS], index=X.index)
            out = df.join(pred_df)

            st.success(f"Predictions generated for {len(out)} rows.")
            st.dataframe(out.head(20), use_container_width=True)

            # Download
            buff2 = io.BytesIO()
            out.to_csv(buff2, index=False)
            st.download_button("Download predictions CSV", buff2.getvalue(),
                               file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

# Footer
st.caption("Tip: If a 'Missing features' message appears, the app lists the exact names it needs. "
           "Headers like 'Temperature_C' and 'Temperature_˚C' are treated as the same during upload.")
