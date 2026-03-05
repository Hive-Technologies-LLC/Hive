# ============================================================
# Hive Battery Analyzer — services/soh.py
# SOH estimation: cycles + temp proxy + resistance proxy + confidence
# ============================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# [SOH:METHODS]
# ------------------------------------------------------------
def soh_from_cycles(df: pd.DataFrame):
    if "cycle_count" not in df.columns:
        return None, {"reason": "missing cycle_count"}
    cc = float(df["cycle_count"].max())
    soh = 100 - (cc / 300) * 10
    return float(np.clip(soh, 0, 100)), {"cycles_max": int(cc)}

def soh_from_temperature_proxy(df: pd.DataFrame):
    if "temp" not in df.columns or df["temp"].dropna().empty:
        return None, {"reason": "missing temp"}
    avg = float(df["temp"].mean())
    soh = 100 - max(0, (avg - 25) * 0.5)
    return float(np.clip(soh, 0, 100)), {"temp_avg_c": round(avg, 1)}

def soh_from_resistance_proxy(df: pd.DataFrame):
    if "voltage" not in df.columns or "current_now" not in df.columns:
        return None, {"reason": "missing fields"}

    v = pd.to_numeric(df["voltage"], errors="coerce")
    i = pd.to_numeric(df["current_now"], errors="coerce")

    if v.dropna().empty or i.dropna().empty or len(v) < 5:
        return None, {"reason": "insufficient samples"}

    dv = v.diff()
    di = i.diff()

    mask = di.abs() > 0.5
    if not mask.any():
        return None, {"reason": "no load steps"}

    R = (dv[mask] / di[mask]).abs().replace([np.inf, -np.inf], np.nan).dropna()
    if R.dropna().empty:
        return None, {"reason": "invalid R"}

    Rm = float(np.median(R))
    R0 = 0.05
    soh = 100 - max(0, (Rm - R0) / 0.005)
    return float(np.clip(soh, 0, 100)), {"R_ohm": round(Rm, 4)}

# ------------------------------------------------------------
# [SOH:CONFIDENCE]
# ------------------------------------------------------------
def soh_confidence(profile: dict) -> dict:
    vc = min(profile["coverage"]["voltage"], profile["coverage"]["current_now"]) / 100
    cycles_cov = profile["coverage"]["cycle_count"] / 100
    temp_cov = profile["coverage"]["temp"] / 100

    sr = profile["median_dt_s"]
    if sr is None:
        sr_score = 0
    elif 0.5 <= sr <= 5:
        sr_score = 1
    elif sr <= 10:
        sr_score = 0.7
    elif sr <= 30:
        sr_score = 0.4
    else:
        sr_score = 0.2

    return {
        "cycles": float(np.clip(cycles_cov, 0, 1)),
        "temp": float(np.clip(temp_cov, 0, 1)),
        "resistance": float(np.clip(0.6 * vc + 0.4 * sr_score, 0, 1))
    }