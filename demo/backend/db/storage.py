# ============================================================
# Hive Battery Analyzer — db/storage.py
# DB init + telemetry logging + admin stats/export
# ============================================================

import io
import sqlite3
import time
import pandas as pd
from fastapi.responses import StreamingResponse

from ..config import DB_FILE, DB_DIR

# ------------------------------------------------------------
# [DB:INIT]
# ------------------------------------------------------------
def init_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            n_rows INTEGER,
            soc REAL,
            soh REAL,
            accuracy_tier TEXT,
            method_soc TEXT,
            method_soh TEXT,
            conf_soc REAL,
            conf_soh REAL,
            temp_avg_c REAL,
            voltage_avg_v REAL,
            current_avg_a REAL
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON telemetry(timestamp);")

# ------------------------------------------------------------
# [DB:LOG]
# ------------------------------------------------------------
def log_telemetry(result: dict, prof: dict, tier: str, stats: dict):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
            INSERT INTO telemetry (
                timestamp, n_rows, soc, soh, accuracy_tier,
                method_soc, method_soh, conf_soc, conf_soh,
                temp_avg_c, voltage_avg_v, current_avg_a
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            prof.get("n_rows"),
            result.get("SOC (%)"),
            result.get("SOH (%)"),
            tier,
            result["methods"]["soc"]["method_used"],
            result["methods"]["soh"]["method_used"],
            result["methods"]["soc"]["confidence"],
            result["methods"]["soh"]["confidence"],
            stats.get("temp_avg_c"),
            stats.get("voltage_avg_v"),
            stats.get("current_avg_a")
        ))

# ------------------------------------------------------------
# [DB:ADMIN_STATS]
# ------------------------------------------------------------
def get_train_stats():
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql("SELECT * FROM telemetry", conn)

    if df.empty:
        return {"records": 0, "message": "No data yet"}

    return {
        "records": int(len(df)),
        "avg_soc": round(float(df["soc"].mean()), 2),
        "avg_soh": round(float(df["soh"].mean()), 2),
        "tiers": df["accuracy_tier"].value_counts().to_dict(),
        "methods_soc": df["method_soc"].value_counts().to_dict(),
        "methods_soh": df["method_soh"].value_counts().to_dict(),
        "avg_conf_soc": round(float(df["conf_soc"].mean()), 3),
        "avg_conf_soh": round(float(df["conf_soh"].mean()), 3)
    }

# ------------------------------------------------------------
# [DB:EXPORT]
# ------------------------------------------------------------
def export_training_csv():
    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql("SELECT * FROM telemetry ORDER BY id ASC", conn)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="telemetry_summary.csv"'}
    )