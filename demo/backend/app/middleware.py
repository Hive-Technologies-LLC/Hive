# ============================================================
# Hive Battery Analyzer — middleware.py
# Upload limit middleware (protects server from huge files)
# ============================================================

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import MAX_BYTES

# ------------------------------------------------------------
# [MW:UPLOAD_LIMIT]
# ------------------------------------------------------------
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/analyze":
            cl = request.headers.get("content-length")
            if cl and int(cl) > MAX_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": "file_too_large", "hint": f"Max {MAX_BYTES//1024//1024} MB"}
                )

            body = await request.body()
            if len(body) > MAX_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": "file_too_large", "hint": f"Max {MAX_BYTES//1024//1024} MB"}
                )

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request._receive = receive

        return await call_next(request)