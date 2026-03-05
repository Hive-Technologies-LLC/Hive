# ============================================================
# Hive Battery Analyzer — services/profile.py
# Profile coverage + sampling rate + rest window detection
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# [PROF:TIME]
# ------------------------------------------------------------
def _to_datetime_series(s):
    if s is None:
        return None
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except:
        return None

def _seconds_between(ts):
    dt = ts.diff().dt.total_seconds()
    return dt.replace([np.inf, -np.inf], np.nan)

# ------------------------------------------------------------
# [PROF:MAIN]
# ------------------------------------------------------------
def profile_data(df: pd.DataFrame) -> dict:
    prof = {
        "n_rows": int(len(df)),
        "fields": list(df.columns),
        "coverage": {
            "voltage": float(df["voltage"].notna().mean()*100) if "voltage" in df.columns else 0,
            "temp": float(df["temp"].notna().mean()*100) if "temp" in df.columns else 0,
            "current_now": float(df["current_now"].notna().mean()*100) if "current_now" in df.columns else 0,
            "cycle_count": float(df["cycle_count"].notna().mean()*100) if "cycle_count" in df.columns else 0,
            "timestamp": 0.0
        },
        "has_rest_window": False,
        "median_dt_s": None
    }

    # Timestamp coverage
    ts = None
    for k in ["timestamp","time","ts","created_at","logged_at"]:
        if k in df.columns:
            ts = _to_datetime_series(df[k])
            break

    if ts is not None and ts.notna().any():
        dt = _seconds_between(ts)
        med = np.nanmedian(dt)
        prof["median_dt_s"] = float(med) if np.isfinite(med) else None
        prof["coverage"]["timestamp"] = float(ts.notna().mean()*100)

    # Rest window detection (near-zero current for >= 3 consecutive samples)
    if "current_now" in df.columns:
        iabs = df["current_now"].abs()
        rest = (iabs <= 0.05).astype(int)
        runs = (rest.groupby((rest != rest.shift()).cumsum()).transform("size") * rest)
        prof["has_rest_window"] = bool((runs >= 3).any())

    return prof