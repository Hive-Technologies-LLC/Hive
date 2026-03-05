/* ============================================================
  HiveOS Lite — API
  File: api.js
  Rev: 0.1.1
============================================================ */

import { API_URL, CONSENT_HEADER } from "./config.js";

export async function analyzeFileWithAPI(file) {
  const fd = new FormData();
  fd.append("file", file);

  // Fail fast if the backend isn't reachable
  const controller = new AbortController();
  const timeoutMs = 12000; // 12 seconds
  const t = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: CONSENT_HEADER,
      body: fd,
      signal: controller.signal
    });
    return res;
  } finally {
    clearTimeout(t);
  }
}