/* ============================================================
  HiveOS Lite — Validators
  File: validators.js
  Rev: 0.1.0
  Purpose:
    Validate user-selected files before API upload.

  QUICK JUMPS (Search These Tags):
    [JS:EXPORTS]
    [JS:MAIN]
============================================================ */

/* [JS:EXPORTS] */
export function isValidFile(file) {
  const allowed = ["application/json", "text/csv", "application/vnd.ms-excel", ""];
  const name = (file?.name || "").toLowerCase();
  const isCSV = name.endsWith(".csv");
  const isJSON = name.endsWith(".json");
  return allowed.includes(file?.type || "") && (isCSV || isJSON);
}

/* [JS:MAIN]
  - We validate by BOTH MIME type + extension.
  - Some systems provide blank MIME types, so "" is allowed.
*/