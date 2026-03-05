/* ============================================================
  HiveOS Lite — File Parse
  File: file_parse.js
  Rev: 0.1.0
  Purpose:
    Read file locally to compute fallback averages (V/I)
    used for UI quick stats when API stats are missing.

  QUICK JUMPS (Search These Tags):
    [JS:EXPORTS]
    [JS:MAIN]
    [JS:NOTES]
============================================================ */

/* [JS:EXPORTS] */
export async function computeAvgVIFromFile(fileObj) {
  const out = { vAvg: null, iAvg: null };

  try {
    const text = await fileObj.text();
    const trimmed = text.trim();

    // JSON path
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      const data = JSON.parse(trimmed);

      const rows = Array.isArray(data) ? data
        : Array.isArray(data.records) ? data.records
        : (typeof data === "object" ? [data] : []);

      const vVals = rows
        .map((r) => Number(r.voltage ?? r.v ?? r.cell_voltage))
        .filter(Number.isFinite);

      const iVals = rows
        .map((r) => Number(r.current_now ?? r.current ?? r.i ?? r.amps ?? r.a))
        .filter(Number.isFinite);

      if (vVals.length) out.vAvg = vVals.reduce((a, b) => a + b, 0) / vVals.length;
      if (iVals.length) out.iAvg = iVals.reduce((a, b) => a + b, 0) / iVals.length;

      return out;
    }

    // CSV path
    const lines = trimmed.split(/\r?\n/).filter(Boolean);
    const hdr = (lines.shift() || "").toLowerCase();
    const cols = hdr.split(",").map((s) => s.trim());

    const idxV = cols.findIndex((c) => ["voltage", "v", "cell_voltage"].includes(c));
    const idxI = cols.findIndex((c) => ["current_now", "current", "i", "amps", "a"].includes(c));

    if (idxV < 0 && idxI < 0) return out;

    let sumV = 0, nV = 0, sumI = 0, nI = 0;

    for (const ln of lines) {
      const parts = ln.split(",");
      if (idxV >= 0) {
        const v = Number(parts[idxV]);
        if (Number.isFinite(v)) { sumV += v; nV++; }
      }
      if (idxI >= 0) {
        const i = Number(parts[idxI]);
        if (Number.isFinite(i)) { sumI += i; nI++; }
      }
    }

    if (nV) out.vAvg = sumV / nV;
    if (nI) out.iAvg = sumI / nI;
  } catch (_) {
    // silent fallback
  }

  return out;
}

/* [JS:NOTES]
  - This is intentionally lightweight, not a full parser.
  - Backend remains the source of truth for results.
*/