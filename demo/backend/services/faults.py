# ============================================================
# Hive Battery Analyzer — services/faults.py
# Simple fault rules (expand later)
# ============================================================

import pandas as pd

def flag_cells(df):
    flags = []
    if df is None or df.empty:
        return flags

    for i, row in df.iterrows():
        v = row.get("voltage")
        t = row.get("temp")

        if pd.notna(v) and v < 3.0:
            flags.append({"Cell ID": int(i), "Issue": "Voltage too low", "Value": float(v)})

        if pd.notna(t) and t > 40:
            flags.append({"Cell ID": int(i), "Issue": "Temperature too high", "Value": float(t)})

    return flags