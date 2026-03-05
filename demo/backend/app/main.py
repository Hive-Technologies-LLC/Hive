# ============================================================
# Hive Battery Analyzer — main.py (v0.2)
# Routes only + middleware wiring + service orchestration
# ============================================================

import shutil
import sys

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from .config import ANALYZER_VERSION, ADMIN_KEY, ALLOWED_ORIGINS, DB_FILE
from .middleware import LimitUploadSizeMiddleware
from .rate_limit import is_rate_limited

from .services.files import is_allowed_file
from .services.analyze import analyze_content

from .db.storage import init_db, log_telemetry, get_train_stats, export_training_csv

# ------------------------------------------------------------
# [APP:SETUP]
# ------------------------------------------------------------
app = FastAPI(title=f"Hive Battery Analyzer {ANALYZER_VERSION}")
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(LimitUploadSizeMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ------------------------------------------------------------
# [APP:STARTUP]
# ------------------------------------------------------------
@app.on_event("startup")
def _startup():
    init_db()

# ------------------------------------------------------------
# [AUTH:HELPERS]
# ------------------------------------------------------------
def _has_admin_access(request: Request) -> bool:
    hdr = request.headers.get("x-admin-key")
    qs = request.query_params.get("key")
    return hdr == ADMIN_KEY or qs == ADMIN_KEY

def _has_consent(request: Request) -> bool:
    raw = (request.headers.get("x-consent") or "").lower().strip()
    return raw in ("1", "true", "yes")

# ------------------------------------------------------------
# [ROUTES:BASIC]
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Hive Battery Analyzer backend running", "version": ANALYZER_VERSION}

@app.get("/health")
def health():
    key_status = "set" if ADMIN_KEY != "dev" else "DEFAULT-DEV"
    return {"status": "ok", "version": ANALYZER_VERSION, "admin_key": key_status}

@app.get("/selftest")
def selftest():
    try:
        disk = shutil.disk_usage(".")
        return {
            "python": sys.version,
            "disk_free_mb": int(disk.free / (1024 * 1024)),
            "db_path": str(DB_FILE),
            "status": "ok"
        }
    except Exception as e:
        return {"error": str(e), "status": "fail"}

# ------------------------------------------------------------
# [ROUTES:ANALYZE]
# ------------------------------------------------------------
@app.post("/analyze")
async def analyze_battery(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks | None = None
):
    ip = request.client.host

    if is_rate_limited(ip):
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "hint": "20 requests / minute max"}
        )

    if not is_allowed_file(file):
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_file", "hint": "Upload .json or .csv"}
        )

    content = await file.read()
    if not content or not content.strip():
        return JSONResponse(status_code=400, content={"error": "empty_file"})

    try:
        result, prof, tier, stats = analyze_content(file.filename, content, ANALYZER_VERSION)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "parse_error", "detail": str(e)})

    # Consent-only telemetry logging
    if _has_consent(request):
        if background_tasks:
            background_tasks.add_task(log_telemetry, result, prof, tier, stats)
        else:
            log_telemetry(result, prof, tier, stats)
    else:
        result["debug"]["consent"] = "not_provided"

    return result

# ------------------------------------------------------------
# [ROUTES:ADMIN]
# ------------------------------------------------------------
@app.get("/train/stats")
def train_stats(request: Request):
    if not _has_admin_access(request):
        raise HTTPException(status_code=401, detail="Admin key required")
    return get_train_stats()

@app.get("/train/export")
def train_export(request: Request):
    if not _has_admin_access(request):
        raise HTTPException(status_code=401, detail="Admin key required")
    return export_training_csv()

# ------------------------------------------------------------
# [RUNNER]
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)