# ============================================================
# Hive Battery Analyzer — rate_limit.py
# In-memory rate limiter (simple + competition-safe)
# ============================================================

import time
from collections import defaultdict

# ------------------------------------------------------------
# [RL:CONFIG]
# ------------------------------------------------------------
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 20
_rate_bucket = defaultdict(list)

# ------------------------------------------------------------
# [RL:API]
# ------------------------------------------------------------
def is_rate_limited(ip: str) -> bool:
    now = time.time()
    bucket = _rate_bucket[ip]
    _rate_bucket[ip] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW]

    if len(_rate_bucket[ip]) >= RATE_LIMIT_MAX:
        return True

    _rate_bucket[ip].append(now)
    return False