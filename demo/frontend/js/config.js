/* ============================================================
  HiveOS Lite — Config
  File: config.js
  Rev: 0.1.1
============================================================ */

/* [JS:EXPORTS] */

// Set this to true for competition/offline demo
export const USE_LOCAL_API = true;

// If local backend is running: python -m uvicorn app.main:app --reload --port 8000
const LOCAL_API = "http://127.0.0.1:8000/analyze";
const PROD_API  = "https://api.hivebatterytools.com/analyze";

export const API_URL = USE_LOCAL_API ? LOCAL_API : PROD_API;

// IMPORTANT: Do NOT send consent by default in client-ready demos
// Make this "no" unless user explicitly opts in
export const CONSENT_HEADER = { "X-Consent": "no" };

export const MAX_RANGE_MI = 25;
export const Q_NOM_AH = 12;