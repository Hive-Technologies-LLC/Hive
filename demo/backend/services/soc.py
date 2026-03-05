/***************************************************************
 *  HiveOS Lite – Backend Service: SOC Estimation
 *  File: services/soc.py
 *  Author: Kashawn Coleman, Hive Technologies
 *  Description:
 *    SOC estimation engine using:
 *      - OCV lookup/interpolation
 *      - Coulomb counting (current over time)
 *      - Confidence scoring per method
 *      - Hybrid blend (confidence-weighted fusion)
 *
 *  Firmware/Service Version: 0.1.1
 *  Build Date: March 2026
 ***************************************************************/
# ============================================================
# Hive Battery Analyzer — services/soc.py
# Rev: 0.1.1
# Purpose:
#   SOC estimation: OCV + Coulomb + confidence scoring + hybrid blend.
#
# Key Constraints (Do Not Break):
#   - Must accept messy telemetry (missing fields are common).
#   - Must return (value, info_dict) for each estimator.
#   - Must never throw on missing columns; return None + reason instead.
#   - All thresholds must be named constants (no magic numbers).
#
# QUICK JUMPS (Search These Tags):
#   [PY:IMPORTS]
#   [PY:CONSTANTS]
#   [PY:OCV-TABLE]
#   [PY:OCV-HELPERS]
#   [PY:SOC-EST-OCV]
#   [PY:SOC-EST-COULOMB]
#   [PY:SOC-CONFIDENCE]
#   [PY:SOC-BLEND]
# ============================================================
#
# Coding Standard Source: Master Coding Standard v1.0  [oai_citation:0‡Master_Coding_Standard_v1.pdf](sediment://file_000000003cac722f88d992e978ed1c6f)

from __future__ import annotations

# ------------------------------------------------------------
# [PY:IMPORTS]
# ------------------------------------------------------------
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .profile import _seconds_between, _to_datetime_series

# ------------------------------------------------------------
# [PY:CONSTANTS]
# ------------------------------------------------------------
# Voltage clamp range for OCV table usage (Li-ion typical)
V_MIN_CLAMP: float = 3.00
V_MAX_CLAMP: float = 4.20

# Coulomb counting default nominal capacity (Ah)
# NOTE: This is a placeholder until you accept capacity input from the user/device metadata.
Q_NOM_AH_DEFAULT: float = 12.0

# Timestamp aliases for cross-vendor logs
TIMESTAMP_KEYS = ("timestamp", "time", "ts", "created_at", "logged_at")

# Confidence scoring thresholds (seconds between samples)
SR_GOOD_MIN_S: float = 0.5
SR_GOOD_MAX_S: float = 5.0
SR_OK_MAX_S: float = 10.0
SR_WEAK_MAX_S: float = 30.0

# Confidence scoring weights (SOC)
W_OCV_VCOV: float = 0.40
W_OCV_REST: float = 0.60
W_OCV_NO_REST: float = 0.20

W_CC_ICOV: float = 0.40
W_CC_TCOV: float = 0.30
W_CC_SR: float = 0.30

# ------------------------------------------------------------
# [PY:OCV-TABLE]
# ------------------------------------------------------------
# OCV → SOC table (simplified). You can replace with chemistry-specific curves later.
# Format: (voltage_V, soc_percent)
OCV_SOC_TABLE = [
    (3.00, 0),
    (3.20, 5),
    (3.40, 15),
    (3.55, 30),
    (3.70, 50),
    (3.85, 70),
    (4.00, 90),
    (4.10, 97),
    (4.20, 100),
]

# ------------------------------------------------------------
# [PY:OCV-HELPERS]
# ------------------------------------------------------------
def soc_from_ocv(v: float) -> float:
    """
    Convert voltage to SOC% using linear interpolation on OCV_SOC_TABLE.

    Why this exists:
      - Provides a usable SOC estimate even when the dataset is minimal.
      - Designed to be robust: clamps voltage and never throws.

    Returns:
      SOC percent in [0, 100].
    """
    v_clamped = float(np.clip(v, V_MIN_CLAMP, V_MAX_CLAMP))

    # Interpolate within the table bounds.
    for (vx, sx), (vy, sy) in zip(OCV_SOC_TABLE, OCV_SOC_TABLE[1:]):
        if vx <= v_clamped <= vy:
            t = (v_clamped - vx) / (vy - vx)
            return float(sx + t * (sy - sx))

    # Fallback: should only happen if table is malformed.
    return float(OCV_SOC_TABLE[-1][1])

def _find_timestamp_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Locate and parse a timestamp-like column into a UTC datetime series.

    Contract:
      - Return None if no timestamp column exists or cannot be parsed.
      - Never throw due to a vendor-specific timestamp format.
    """
    for k in TIMESTAMP_KEYS:
        if k in df.columns:
            ts = _to_datetime_series(df[k])
            return ts
    return None

# ------------------------------------------------------------
# [PY:SOC-EST-OCV]
# ------------------------------------------------------------
def estimate_soc_ocv(df: pd.DataFrame) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    SOC estimator #1: OCV-based SOC.

    Requirements:
      - Needs: voltage
      - Works best when the pack is at rest (no/low current).
    """
    if df is None or df.empty:
        return None, {"reason": "empty dataframe"}

    if "voltage" not in df.columns or df["voltage"].dropna().empty:
        return None, {"reason": "missing voltage"}

    vavg = float(pd.to_numeric(df["voltage"], errors="coerce").mean())
    if not np.isfinite(vavg):
        return None, {"reason": "invalid voltage"}

    soc = float(np.clip(soc_from_ocv(vavg), 0.0, 100.0))
    return soc, {"v_avg": round(vavg, 3), "v_clamp": [V_MIN_CLAMP, V_MAX_CLAMP]}

# ------------------------------------------------------------
# [PY:SOC-EST-COULOMB]
# ------------------------------------------------------------
def estimate_soc_coulomb(
    df: pd.DataFrame,
    q_nom_ah: float = Q_NOM_AH_DEFAULT
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    SOC estimator #2: Coulomb counting (CC).

    High-level:
      - Integrate current over time to estimate Ah moved.
      - Apply that delta against a nominal capacity to adjust SOC.

    Requirements:
      - Needs: current_now + timestamp (at least 2 valid timestamps)
      - Assumes q_nom_ah is representative (placeholder until user/device provides it)

    Returns:
      (soc_percent or None, info_dict)
    """
    if df is None or df.empty:
        return None, {"reason": "empty dataframe"}

    if "current_now" not in df.columns:
        return None, {"reason": "missing current"}

    ts = _find_timestamp_series(df)
    if ts is None or ts.notna().sum() < 2:
        return None, {"reason": "missing timestamp"}

    dt_s = _seconds_between(ts)  # seconds between samples (may include NaN)
    I_a = pd.to_numeric(df["current_now"], errors="coerce")

    if I_a.dropna().empty or dt_s.dropna().empty:
        return None, {"reason": "bad current/time"}

    # Integrate current over time in hours: sum(I * dt) / 3600
    dAh = float((I_a.fillna(0.0) * dt_s.fillna(0.0)).sum() / 3600.0)

    # Base SOC: prefer OCV if voltage exists; else neutral 50%.
    base_soc, _ = estimate_soc_ocv(df) if "voltage" in df.columns else (50.0, {})
    if base_soc is None:
        base_soc = 50.0

    if not (np.isfinite(q_nom_ah) and q_nom_ah > 0):
        return None, {"reason": "invalid q_nom_ah"}

    soc = float(base_soc - (dAh / q_nom_ah) * 100.0)
    soc = float(np.clip(soc, 0.0, 100.0))

    return soc, {
        "base_soc": round(float(base_soc), 1),
        "dAh": round(dAh, 3),
        "q_nom_ah": float(q_nom_ah),
    }

# ------------------------------------------------------------
# [PY:SOC-CONFIDENCE]
# ------------------------------------------------------------
def soc_confidence(profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute confidence scores (0..1) for each SOC method.

    WHY this exists (not WHAT):
      - We want the system to adapt automatically to bad/missing telemetry.
      - Confidence weights allow a safe hybrid blend rather than hard switching.

    Expected profile contract (from services/profile.py):
      profile["coverage"]["voltage"]      in [0..100]
      profile["coverage"]["current_now"]  in [0..100]
      profile["coverage"]["timestamp"]    in [0..100]
      profile["has_rest_window"]          bool
      profile["median_dt_s"]              float|None
    """
    cov = profile.get("coverage", {}) or {}

    v_cov = float(cov.get("voltage", 0.0)) / 100.0
    i_cov = float(cov.get("current_now", 0.0)) / 100.0
    t_cov = float(cov.get("timestamp", 0.0)) / 100.0

    # OCV confidence: voltage coverage + presence of a rest window.
    # Rationale: OCV is only trustworthy when the pack is near rest.
    ocv_rest_bonus = W_OCV_REST if bool(profile.get("has_rest_window")) else W_OCV_NO_REST
    c_ocv = (W_OCV_VCOV * v_cov) + ocv_rest_bonus

    # Sample-rate score for Coulomb counting: stable dt improves integration quality.
    sr = profile.get("median_dt_s", None)
    if sr is None or not np.isfinite(sr):
        sr_score = 0.0
    elif SR_GOOD_MIN_S <= sr <= SR_GOOD_MAX_S:
        sr_score = 1.0
    elif sr <= SR_OK_MAX_S:
        sr_score = 0.6
    elif sr <= SR_WEAK_MAX_S:
        sr_score = 0.3
    else:
        sr_score = 0.1

    # CC confidence: needs current + timestamp + acceptable sampling.
    c_cc = (W_CC_ICOV * i_cov) + (W_CC_TCOV * t_cov) + (W_CC_SR * sr_score)

    return {
        "ocv": float(np.clip(c_ocv, 0.0, 1.0)),
        "cc": float(np.clip(c_cc, 0.0, 1.0)),
    }

# ------------------------------------------------------------
# [PY:SOC-BLEND]
# ------------------------------------------------------------
def blend_soc(
    s1: Optional[float],
    w1: float,
    s2: Optional[float],
    w2: float
) -> Optional[float]:
    """
    Confidence-weighted SOC fusion.

    Rules:
      - If one estimate is missing, return the other.
      - If both are missing, return None.
      - If weights are zero-ish, fall back to simple average (safe + deterministic).
    """
    if s1 is None and s2 is None:
        return None
    if s1 is None:
        return float(s2)
    if s2 is None:
        return float(s1)

    denom = float(w1 + w2)
    if denom < 1e-6:
        return float(0.5 * (float(s1) + float(s2)))

    return float((w1 * float(s1) + w2 * float(s2)) / denom)