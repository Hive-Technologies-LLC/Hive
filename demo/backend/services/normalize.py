# ============================================================
# Hive Battery Analyzer — services/normalize.py
# Parse + normalize incoming CSV/JSON to a clean DataFrame
# ============================================================

import io, json
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# [NORM:ALIASES]
# ------------------------------------------------------------
ALIAS_MAP = {
    "voltage": ["voltage","v","volts","cell_voltage","mv","millivolts"],
    "temp": ["temp","temperature","temp_c","temperature_c","tempf","temperature_f","t"],
    "current_now": ["current_now","current","i","amps","a","ma","milliamps","milliamperes"],
    "cycle_count": ["cycle_count","cycles","cycle","charge_cycles"]
}

# ------------------------------------------------------------
# [NORM:COLS]
# ------------------------------------------------------------
def _clean_colname(name: str) -> str:
    return str(name).strip().lower().replace(" ","_").replace(".", "_")

def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [_clean_colname(c) for c in df.columns]
    for canon, aliases in ALIAS_MAP.items():
        if canon not in df.columns:
            for a in aliases:
                a_clean = _clean_colname(a)
                if a_clean in df.columns:
                    df.rename(columns={a_clean: canon}, inplace=True)
                    break
    return df

def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["voltage","temp","current_now","cycle_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------------------------------------------------
# [NORM:UNITS]
# ------------------------------------------------------------
def _convert_units(df: pd.DataFrame, units: dict | None) -> pd.DataFrame:
    units = {k.lower(): (v.lower() if isinstance(v,str) else v) for k,v in (units or {}).items()}

    # voltage
    if "voltage" in df.columns:
        if units.get("voltage") in ["mv","millivolts"]:
            df["voltage"] /= 1000.0
        else:
            med = pd.to_numeric(df["voltage"], errors="coerce").median()
            if pd.notna(med) and med > 10:
                df["voltage"] /= 1000.0

    # temp
    if "temp" in df.columns:
        if units.get("temp") in ["f","fahrenheit","degf","°f"]:
            df["temp"] = (df["temp"] - 32) * 5/9
        else:
            med = pd.to_numeric(df["temp"], errors="coerce").median()
            if pd.notna(med) and med > 80:
                df["temp"] = (df["temp"] - 32) * 5/9

    # current
    if "current_now" in df.columns:
        if units.get("current_now") in ["ma","milliamps","milliamperes"]:
            df["current_now"] /= 1000.0
        else:
            med = pd.to_numeric(df["current_now"], errors="coerce").median()
            if pd.notna(med) and med > 50:
                df["current_now"] /= 1000.0

    return df

# ------------------------------------------------------------
# [NORM:JSON]
# ------------------------------------------------------------
def _parse_json_bytes(b: bytes):
    text = b.decode("utf-8").strip()
    units = None

    # Normal JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            units = data.get("units") or data.get("Units")
            for key in ["records","data","rows","items"]:
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key])
                    break
            else:
                if any(isinstance(v,list) for v in data.values()):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize([data])
        else:
            raise ValueError("Unsupported JSON")

        df.columns = [_clean_colname(c) for c in df.columns]
        return df, units
    except json.JSONDecodeError:
        pass

    # JSON lines fallback
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty JSON")

    objs, units = [], None
    for ln in lines:
        if ln.endswith(","):
            ln = ln[:-1]
        obj = json.loads(ln)
        if isinstance(obj, dict) and "units" in obj and units is None:
            units = obj["units"]
            if isinstance(obj.get("records"), list):
                objs.extend(obj["records"])
            continue
        if isinstance(obj, dict) and isinstance(obj.get("records"), list):
            objs.extend(obj["records"])
            continue
        objs.append(obj)

    df = pd.json_normalize(objs)
    df.columns = [_clean_colname(c) for c in df.columns]
    return df, units

# ------------------------------------------------------------
# [NORM:ENTRY]
# ------------------------------------------------------------
def normalize_dataframe_from_upload(filename: str, content: bytes):
    units = None

    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [_clean_colname(c) for c in df.columns]
    elif filename.endswith(".json"):
        df, units = _parse_json_bytes(content)
    else:
        raise ValueError("Unsupported file")

    df = _rename_aliases(df)
    df = _coerce_numeric(df)
    df = _convert_units(df, units)

    debug = {"columns_after": list(df.columns), "units_hint": units or {}}
    return df, debug