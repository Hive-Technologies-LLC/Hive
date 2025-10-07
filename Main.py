# main.py
import os, io, json
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

ANALYZER_VERSION = "v0.1"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB upload cap (adjust as needed)

app = FastAPI(title="Hive Battery Analyzer")

# -------------------- Upload size guard (413 on oversize) --------------------
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/analyze":
            # Quick header check (may be missing if chunked)
            cl = request.headers.get("content-length")
            if cl and int(cl) > MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_BYTES//1024//1024} MB).")

            # Read body once; re-inject for downstream handlers
            body = await request.body()
            if len(body) > MAX_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_BYTES//1024//1024} MB).")

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request._receive = receive

        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware)

# -------------------- CORS (loose for dev; tighten for prod) --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: set to ["https://yourdomain.com"] for prod
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------- Basic routes --------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": ANALYZER_VERSION}

# Serve index.html from same folder (MVP convenience)
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """<!DOCTYPE html><html><body style="font-family:sans-serif">
        <h2>Hive Battery Analyzer</h2>
        <p>Upload page not found. Place <code>index.html</code> next to <code>main.py</code>.</p>
        </body></html>"""

# -------------------- Normalization helpers --------------------
ALIAS_MAP = {
    "voltage":     ["voltage", "v", "volts", "cell_voltage", "mv", "millivolts"],
    "temp":        ["temp", "temperature", "temp_c", "temperature_c", "tempf", "temperature_f", "t"],
    "current_now": ["current_now", "current", "i", "amps", "a", "ma", "milliamps", "milliamperes"],
    "cycle_count": ["cycle_count", "cycles", "cycle", "charge_cycles"]
}

def _clean_colname(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace(".", "_")

def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_clean_colname(c) for c in df.columns]
    for canon, aliases in ALIAS_MAP.items():
        if canon in df.columns:
            continue
        for a in aliases:
            a_clean = _clean_colname(a)
            if a_clean in df.columns and canon not in df.columns:
                df.rename(columns={a_clean: canon}, inplace=True)
                break
    return df

def _convert_units(df: pd.DataFrame, units: dict | None) -> pd.DataFrame:
    """Convert common units to V / °C / A using explicit units or heuristics."""
    units = {k.lower(): (v.lower() if isinstance(v, str) else v) for k, v in (units or {}).items()}

    # Voltage: mV -> V or heuristic if values look like mV
    if "voltage" in df.columns:
        if units.get("voltage") in ["mv", "millivolts"]:
            df["voltage"] = df["voltage"] / 1000.0
        else:
            med = pd.to_numeric(df["voltage"], errors="coerce").median()
            if pd.notna(med) and med > 10:  # cell V should be < 10
                df["voltage"] = df["voltage"] / 1000.0

    # Temperature: F -> C or heuristic if values look like °F
    if "temp" in df.columns:
        if units.get("temp") in ["f", "fahrenheit", "degf", "°f"]:
            df["temp"] = (df["temp"] - 32.0) * 5.0 / 9.0
        else:
            med = pd.to_numeric(df["temp"], errors="coerce").median()
            if pd.notna(med) and med > 80:  # likely °F
                df["temp"] = (df["temp"] - 32.0) * 5.0 / 9.0

    # Current: mA -> A or heuristic if numbers are very large
    if "current_now" in df.columns:
        if units.get("current_now") in ["ma", "milliamps", "milliamperes"]:
            df["current_now"] = df["current_now"] / 1000.0
        else:
            med = pd.to_numeric(df["current_now"], errors="coerce").median()
            if pd.notna(med) and med > 50:  # many telemetry feeds send mA
                df["current_now"] = df["current_now"] / 1000.0
    return df

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["voltage", "temp", "current_now", "cycle_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _parse_json_bytes(b: bytes) -> tuple[pd.DataFrame, dict | None]:
    """
    Accepts:
      Standard JSON:
        - [ {...}, {...} ]                      (array of rows)
        - {"records":[...], "units": {...}}     (wrapped rows + optional units)
        - {"data":[...]}, {"rows":[...]}, {"items":[...]}
        - {"voltage":[...], "temp":[...]}       (columnar dict)
        - {"voltage":3.9, "temp":30.1, ...}     (single row)
      JSON Lines (NDJSON):
        - One JSON object per line, optional first metadata line with {"units": {...}} or {"records":[...]}
    Returns: (DataFrame, units_dict_or_None)
    """
    text = b.decode("utf-8").strip()
    units = None

    # First try: standard JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            units = data.get("units") or data.get("Units")
            for key in ["records", "data", "rows", "items"]:
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key])
                    break
            else:
                if any(isinstance(v, list) for v in data.values()):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize([data])
        else:
            raise ValueError("Unsupported JSON shape")
        df.columns = [_clean_colname(c) for c in df.columns]
        return df, units
    except json.JSONDecodeError:
        pass

    # Fallback: NDJSON (one JSON object per line)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty JSON payload")

    objs: list[dict] = []
    for ln in lines:
        if ln.endswith(","):
            ln = ln[:-1]
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON/JSONL near: {ln[:80]} ... ({e})")
        if isinstance(obj, dict) and "units" in obj and units is None:
            units = obj.get("units")
            if isinstance(obj.get("records"), list):
                objs.extend(obj["records"])
            continue
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            objs.extend(obj["records"])
            continue
        objs.append(obj)

    if not objs:
        raise ValueError("No rows found in JSON Lines input")

    df = pd.json_normalize(objs)
    df.columns = [_clean_colname(c) for c in df.columns]
    return df, units

def normalize_dataframe_from_upload(file: UploadFile, content: bytes) -> tuple[pd.DataFrame, dict]:
    """Read CSV/JSON, flatten, alias-rename, coerce numerics, convert units."""
    units = None
    if file.filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [_clean_colname(c) for c in df.columns]
    elif file.filename.endswith(".json"):
        df, units = _parse_json_bytes(content)
    else:
        raise ValueError("Unsupported file type. Use .csv or .json")

    df = _rename_aliases(df)
    df = _coerce_numeric(df)
    df = _convert_units(df, units)

    debug = {"columns_after": list(df.columns), "units_hint": units or {}}
    return df, debug

# -------------------- Small utils & profiling --------------------
def _to_datetime_series(s):
    if s is None:
        return None
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return None

def _seconds_between(ts):
    dt = ts.diff().dt.total_seconds()
    return dt.replace([np.inf, -np.inf], np.nan)

def _has_columns(df, cols):
    return all(c in df.columns for c in cols)

def profile_data(df: pd.DataFrame):
    """Quick facts used to score method confidence."""
    prof = {
        "n_rows": int(len(df)),
        "fields": list(df.columns),
        "coverage": {
            "voltage": float(df["voltage"].notna().mean()*100) if "voltage" in df.columns else 0.0,
            "temp": float(df["temp"].notna().mean()*100) if "temp" in df.columns else 0.0,
            "current_now": float(df["current_now"].notna().mean()*100) if "current_now" in df.columns else 0.0,
            "cycle_count": float(df["cycle_count"].notna().mean()*100) if "cycle_count" in df.columns else 0.0,
            "timestamp": 0.0
        },
        "has_rest_window": False,
        "median_dt_s": None
    }

    # Timestamp coverage & median dt
    ts = None
    for k in ["timestamp", "time", "ts", "created_at", "logged_at"]:
        if k in df.columns:
            ts = _to_datetime_series(df[k])
            if ts is not None:
                break

    if ts is not None and ts.notna().any():
        dt = _seconds_between(ts)
        med = np.nanmedian(dt)
        prof["median_dt_s"] = float(med) if np.isfinite(med) else None
        prof["coverage"]["timestamp"] = float(ts.notna().mean()*100)

    # Detect "rest" (|I| < 0.05A) windows of >= 3 samples
    if "current_now" in df.columns:
        iabs = df["current_now"].abs()
        rest = (iabs <= 0.05).astype(int)
        runs = (rest.groupby((rest != rest.shift()).cumsum()).transform('size') * rest)
        prof["has_rest_window"] = bool((runs >= 3).any())

    return prof

# -------------------- SOC ensemble --------------------
OCV_SOC_TABLE = [
    (3.00, 0.0), (3.20, 5.0), (3.40, 15.0), (3.55, 30.0),
    (3.70, 50.0), (3.85, 70.0), (4.00, 90.0), (4.10, 97.0), (4.20, 100.0),
]

def soc_from_ocv(voltage_v):
    v = float(np.clip(voltage_v, OCV_SOC_TABLE[0][0], OCV_SOC_TABLE[-1][0]))
    for (vx, sx), (vy, sy) in zip(OCV_SOC_TABLE, OCV_SOC_TABLE[1:]):
        if vx <= v <= vy:
            t = (v - vx) / (vy - vx) if vy != vx else 0.0
            return sx + t * (sy - sx)
    return OCV_SOC_TABLE[-1][1]

def estimate_soc_ocv(df):
    if "voltage" not in df.columns or df["voltage"].dropna().empty:
        return None, {"reason": "missing voltage"}
    v_avg = float(df["voltage"].mean(skipna=True))
    soc = soc_from_ocv(v_avg)
    return float(np.clip(soc, 0, 100)), {"v_avg": round(v_avg,3)}

def estimate_soc_coulomb(df):
    # Need current + timestamp
    if "current_now" not in df.columns:
        return None, {"reason": "missing current"}
    ts = None
    for k in ["timestamp", "time", "ts", "created_at", "logged_at"]:
        if k in df.columns:
            ts = _to_datetime_series(df[k])
            if ts is not None and ts.notna().any():
                break
    if ts is None:
        return None, {"reason": "missing timestamp"}

    dt = _seconds_between(ts)
    i = pd.to_numeric(df["current_now"], errors="coerce")
    if i.dropna().empty or dt.dropna().empty:
        return None, {"reason": "insufficient current/time data"}

    Q_nom_Ah = 12.0  # tweak per vehicle
    dAh = (i.fillna(0) * dt.fillna(0)).sum() / 3600.0
    base_soc, _ = estimate_soc_ocv(df) if "voltage" in df.columns else (50.0, {})
    soc = base_soc - (dAh / Q_nom_Ah) * 100.0
    return float(np.clip(soc, 0, 100)), {
        "base_soc": round(base_soc,1),
        "dAh": round(float(dAh),3),
        "Q_nom_Ah": Q_nom_Ah
    }

def blend_soc(s_ocv, conf_ocv, s_cc, conf_cc):
    if s_ocv is None and s_cc is None:
        return None
    if s_ocv is None: return s_cc
    if s_cc  is None: return s_ocv
    w1 = max(conf_ocv, 0.0); w2 = max(conf_cc, 0.0)
    if (w1 + w2) <= 1e-6:
        return 0.5*s_ocv + 0.5*s_cc
    return float((w1*s_ocv + w2*s_cc) / (w1 + w2))

def soc_confidence(profile):
    c = {"ocv": 0.0, "cc": 0.0}
    # OCV: voltage coverage + rest window bonus
    v_cov = profile["coverage"]["voltage"]/100.0
    c["ocv"] = 0.4*v_cov + (0.6 if profile["has_rest_window"] else 0.2)

    # CC: current + timestamp coverage + sampling quality
    t_cov = profile["coverage"]["timestamp"]/100.0
    i_cov = profile["coverage"]["current_now"]/100.0
    sr = profile["median_dt_s"]
    if sr is None: sr_score = 0.0
    elif 0.5 <= sr <= 5.0: sr_score = 1.0
    elif sr <= 10: sr_score = 0.6
    elif sr <= 30: sr_score = 0.3
    else: sr_score = 0.1
    c["cc"] = 0.4*i_cov + 0.3*t_cov + 0.3*sr_score

    c["ocv"] = float(np.clip(c["ocv"], 0, 1))
    c["cc"]  = float(np.clip(c["cc"],  0, 1))
    return c

# -------------------- SOH ensemble --------------------
def soh_from_cycles(df):
    if "cycle_count" not in df.columns or df["cycle_count"].dropna().empty:
        return None, {"reason":"missing cycle_count"}
    cc = float(df["cycle_count"].max())
    soh = 100.0 - (cc/300.0)*10.0   # simple heuristic
    return float(np.clip(soh, 0, 100)), {"cycles_max": int(cc)}

def soh_from_temperature_proxy(df):
    if "temp" not in df.columns or df["temp"].dropna().empty:
        return None, {"reason":"missing temp"}
    temp_avg = float(df["temp"].mean(skipna=True))
    soh = 100 - max(0.0, (temp_avg - 25.0) * 0.5)
    return float(np.clip(soh, 0, 100)), {"temp_avg_c": round(temp_avg,1)}

def soh_from_resistance_proxy(df):
    if not _has_columns(df, ["voltage", "current_now"]):
        return None, {"reason":"missing voltage/current"}
    v = pd.to_numeric(df["voltage"], errors="coerce")
    i = pd.to_numeric(df["current_now"], errors="coerce")
    if v.dropna().empty or i.dropna().empty or len(v) < 5:
        return None, {"reason":"insufficient samples"}

    dv = v.diff()
    di = i.diff()
    mask = di.abs() > 0.5  # load steps
    if not mask.any():
        return None, {"reason":"no load steps detected"}

    R = np.median((dv[mask] / di[mask]).abs().replace([np.inf,-np.inf], np.nan).dropna())  # ohms
    if not np.isfinite(R):
        return None, {"reason":"invalid R"}

    R0 = 0.05
    soh = 100.0 - max(0.0, (R - R0)/0.005)*1.0
    return float(np.clip(soh, 0, 100)), {"R_ohm": round(float(R),4)}

def soh_confidence(profile):
    c = {"cycles":0.0, "temp":0.0, "resistance":0.0}
    c["cycles"] = float(np.clip(profile["coverage"]["cycle_count"]/100.0, 0, 1))
    c["temp"] = float(np.clip(profile["coverage"]["temp"]/100.0, 0, 1))
    vc = min(profile["coverage"]["voltage"], profile["coverage"]["current_now"])/100.0
    sr = profile["median_dt_s"]
    if sr is None: sr_score = 0.0
    elif 0.5 <= sr <= 5.0: sr_score = 1.0
    elif sr <= 10: sr_score = 0.7
    elif sr <= 30: sr_score = 0.4
    else: sr_score = 0.2
    c["resistance"] = float(np.clip(0.6*vc + 0.4*sr_score, 0, 1))
    return c

# -------------------- Existing simple rules --------------------
def soc_from_voltage(df: pd.DataFrame):
    if "voltage" not in df.columns or df["voltage"].dropna().empty:
        return None
    v_avg = df["voltage"].mean(skipna=True)
    soc = (v_avg - 3.0) / (4.2 - 3.0) * 100
    return round(max(0, min(soc, 100)), 2)

def soh_from_temperature(df: pd.DataFrame):
    if "temp" not in df.columns or df["temp"].dropna().empty:
        return None
    temp_avg = df["temp"].mean(skipna=True)
    soh = 100 - max(0, (temp_avg - 25) * 0.5)
    return round(max(0, min(soh, 100)), 2)

def flag_cells(df: pd.DataFrame):
    flagged = []
    if df.empty:
        return flagged
    for i, row in df.iterrows():
        v = row.get("voltage", None)
        t = row.get("temp", None)
        if pd.notna(v) and v < 3.0:
            flagged.append({"Cell ID": int(i), "Issue": "Voltage too low", "Value": float(v)})
        if pd.notna(t) and t > 40:
            flagged.append({"Cell ID": int(i), "Issue": "Temperature too high", "Value": float(t)})
    return flagged

def score_data_quality(df: pd.DataFrame):
    fields = ["voltage", "temp", "current_now", "cycle_count"]
    score = sum(1 for f in fields if f in df.columns)
    if score >= 4:
        return "Tier 3 – Full"
    elif score == 3:
        return "Tier 2 – Enhanced"
    elif score == 2:
        return "Tier 1 – Basic"
    else:
        return "Tier 0 – Minimal"

# -------------------- Server-side quick stats (for UI right column) --------------------
def basic_stats(df: pd.DataFrame):
    def mean_or_none(col):
        return None if col not in df.columns or df[col].dropna().empty else float(df[col].mean(skipna=True))
    return {
        "voltage_avg_v": mean_or_none("voltage"),
        "current_avg_a": mean_or_none("current_now"),
        "temp_avg_c":    mean_or_none("temp"),
    }

# -------------------- API: /analyze --------------------
@app.post("/analyze")
async def analyze_battery(file: UploadFile = File(...)):
    content = await file.read()

    try:
        df, debug = normalize_dataframe_from_upload(file, content)
    except HTTPException as e:
        # Re-raise HTTPExceptions (e.g., 413) so status code is preserved
        raise e
    except Exception as e:
        return {
            "error": f"{e}",
            "version": ANALYZER_VERSION,
            "SOC (%)": None,
            "SOH (%)": None,
            "Flagged Cells": [],
            "Accuracy Tier": "Tier 0 – Minimal",
            "methods": {"soc": {}, "soh": {}},
            "stats": {"voltage_avg_v": None, "current_avg_a": None, "temp_avg_c": None},
            "errors": [str(e)],
            "debug": {}
        }

    if not df.empty:
        df = df.dropna(how="all")

    # Profile dataset
    prof = profile_data(df)

    # SOC ensemble
    soc_ocv, info_ocv = estimate_soc_ocv(df)
    soc_cc,  info_cc  = estimate_soc_coulomb(df)
    conf_soc = soc_confidence(prof)
    soc_hybrid = blend_soc(soc_ocv, conf_soc["ocv"], soc_cc, conf_soc["cc"])

    soc_candidates = []
    if soc_ocv is not None: soc_candidates.append(("OCV", soc_ocv, conf_soc["ocv"], info_ocv))
    if soc_cc  is not None: soc_candidates.append(("Coulomb", soc_cc, conf_soc["cc"], info_cc))
    if soc_hybrid is not None:
        soc_candidates.append(("Hybrid", soc_hybrid, max(conf_soc["ocv"], conf_soc["cc"]), {"blend":"weighted"}))
    if soc_candidates:
        soc_method, soc_value, soc_conf_best, soc_reason = max(soc_candidates, key=lambda x: x[2])
    else:
        soc_method, soc_value, soc_conf_best, soc_reason = (None, None, 0.0, {"reason":"no usable SOC method"})

    # SOH ensemble
    soh_cyc,  cyc_info  = soh_from_cycles(df)
    soh_tmp,  tmp_info  = soh_from_temperature_proxy(df)
    soh_res,  res_info  = soh_from_resistance_proxy(df)
    conf_soh = soh_confidence(prof)

    soh_candidates = []
    if soh_cyc is not None: soh_candidates.append(("Cycles", soh_cyc, conf_soh["cycles"], cyc_info))
    if soh_tmp is not None: soh_candidates.append(("TempProxy", soh_tmp, conf_soh["temp"], tmp_info))
    if soh_res is not None: soh_candidates.append(("Resistance", soh_res, conf_soh["resistance"], res_info))
    if soh_candidates:
        soh_method, soh_value, soh_conf_best, soh_reason = max(soh_candidates, key=lambda x: x[2])
    else:
        soh_method, soh_value, soh_conf_best, soh_reason = (None, None, 0.0, {"reason":"no usable SOH method"})

    # Existing flags + tier
    flags = flag_cells(df)
    tier  = score_data_quality(df)

    # Server-side quick stats
    stats = basic_stats(df)

    # Deterministic response
    result = {
        "version": ANALYZER_VERSION,
        "SOC (%)": round(soc_value, 2) if soc_value is not None else None,
        "SOH (%)": round(soh_value, 2) if soh_value is not None else None,
        "Flagged Cells": flags,
        "Accuracy Tier": tier,
        "methods": {
            "soc": {
                "method_used": soc_method,
                "confidence": round(float(soc_conf_best), 3),
                "candidates": {
                    "OCV":     {"value": None if soc_ocv    is None else round(soc_ocv,2),    "confidence": round(conf_soc["ocv"],3), "info": info_ocv},
                    "Coulomb": {"value": None if soc_cc     is None else round(soc_cc,2),     "confidence": round(conf_soc["cc"],3),  "info": info_cc},
                    "Hybrid":  {"value": None if soc_hybrid is None else round(soc_hybrid,2), "confidence": round(max(conf_soc["ocv"], conf_soc["cc"]),3), "info": {"blend":"weighted"}}
                }
            },
            "soh": {
                "method_used": soh_method,
                "confidence": round(float(soh_conf_best), 3),
                "candidates": {
                    "Cycles":     {"value": None if soh_cyc is None else round(soh_cyc,2), "confidence": round(conf_soh["cycles"],3),     "info": cyc_info},
                    "TempProxy":  {"value": None if soh_tmp is None else round(soh_tmp,2), "confidence": round(conf_soh["temp"],3),       "info": tmp_info},
                    "Resistance": {"value": None if soh_res is None else round(soh_res,2), "confidence": round(conf_soh["resistance"],3), "info": res_info}
                }
            }
        },
        "stats": stats,        # <- new for UI right column
        "errors": [],          # keep array for future per-row validations
        "debug": {
            "profile": prof,
            "columns_after": debug.get("columns_after", []),
            "units_hint": debug.get("units_hint", {})
        }
    }
    return result

# -------------------- Runner --------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
