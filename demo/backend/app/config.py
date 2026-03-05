# ============================================================
# Hive Battery Analyzer — config.py
# Centralized env vars + constants
# ============================================================

import os
from pathlib import Path

# ------------------------------------------------------------
# [CFG:PATHS] Base paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
APP_DIR  = Path(__file__).resolve().parent         # backend/app/

# ------------------------------------------------------------
# [CFG:CORE] Version + runtime
# ------------------------------------------------------------
ANALYZER_VERSION = os.getenv("ANALYZER_VERSION", "v0.2")
MAX_BYTES = int(os.getenv("MAX_BYTES", 10 * 1024 * 1024))  # 10MB default

# ------------------------------------------------------------
# [CFG:SECURITY] Admin key (must change later)
# ------------------------------------------------------------
ADMIN_KEY = os.getenv("ADMIN_KEY", "dev")

# ------------------------------------------------------------
# [CFG:CORS] Allowed origins
# ------------------------------------------------------------
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://hiveos-lite.vercel.app,http://localhost:8000"
).split(",")

# ------------------------------------------------------------
# [CFG:DB] DB location (DB correction)
# Default to backend/data/telemetry_summary.db
# ------------------------------------------------------------
DEFAULT_DB_PATH = BASE_DIR / "data" / "telemetry_summary.db"
DB_FILE = Path(os.getenv("DB_FILE", str(DEFAULT_DB_PATH)))
DB_DIR = DB_FILE.parent