# ============================================================
# Hive Battery Analyzer — services/files.py
# File validation helpers
# ============================================================

import os
from fastapi import UploadFile

# ------------------------------------------------------------
# [FILES:VALIDATION]
# ------------------------------------------------------------
def is_allowed_file(file: UploadFile) -> bool:
    name = (file.filename or "").lower()
    ext = os.path.splitext(name)[1]

    good_ext = {".json", ".csv"}
    good_mime = {
        "application/json",
        "text/csv",
        "application/vnd.ms-excel",
        ""
    }
    return ext in good_ext and (file.content_type in good_mime)