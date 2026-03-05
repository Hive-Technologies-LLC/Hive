/* ============================================================
  HiveOS Lite — Renderers
  File: render.js
  Rev: 0.1.0
  Purpose:
    UI render functions:
      - SOC quick stats
      - Fault list
      - SOH battery bar fill

  QUICK JUMPS (Search These Tags):
    [JS:IMPORTS]
    [JS:SOC-STATS]
    [JS:FAULTS]
    [JS:SOH]
	 [JS:CONFIDENCE]
============================================================ */

/* [JS:IMPORTS] */
import { MAX_RANGE_MI, Q_NOM_AH } from "./config.js";

/* [JS:SOC-STATS] */
export function populateSocStats(ui, apiData, socPct, vAvgFallback, iAvgFallback) {
  const apiStats = apiData?.stats || {};

  const vAvg = Number.isFinite(apiStats.voltage_avg_v) ? apiStats.voltage_avg_v : vAvgFallback;
  const iAvg = Number.isFinite(apiStats.current_avg_a) ? apiStats.current_avg_a : iAvgFallback;

  const fmt = (x, u) => (x == null || !Number.isFinite(x)) ? "—" : `${Math.round(x * 10) / 10}${u}`;

  const estRange = (socPct == null || !Number.isFinite(socPct)) ? null : (socPct / 100) * MAX_RANGE_MI;

  let ttfMin = null;
  if (Number.isFinite(socPct) && Number.isFinite(iAvg) && iAvg > 0) {
    const remainingAh = ((100 - socPct) / 100) * Q_NOM_AH;
    ttfMin = (remainingAh / iAvg) * 60;
  }

  ui.statVoltage.textContent = fmt(vAvg, " V");
  ui.statCurrent.textContent = fmt(iAvg, " A");
  ui.statRange.textContent = fmt(estRange, " mi");
  ui.statTTF.textContent =
    (ttfMin == null || !Number.isFinite(ttfMin)) ? "—" :
    (ttfMin >= 60 ? `${Math.round(ttfMin / 60)}h` : `${Math.round(ttfMin)}m`);
}

/* [JS:FAULTS] */
export function renderFaults(ui, flags) {
  ui.faultList.innerHTML = "";

  if (!flags || !flags.length) {
    ui.faultList.innerHTML = '<div class="item"><span class="mutedSmall">No issues detected.</span></div>';
    return;
  }

  flags.slice(0, 6).forEach((f) => {
    const html = `<div class="item">
      <div>
        <strong>${f.Issue || "Fault"}</strong>
        <div class="mutedSmall">
          ${f["Cell ID"] != null ? "Cell " + f["Cell ID"] : "—"} • Value: ${f.Value ?? "—"}
        </div>
      </div>
      <div class="mutedSmall">now</div>
    </div>`;
    ui.faultList.insertAdjacentHTML("beforeend", html);
  });
}

/* [JS:SOH] */
export function renderBatteryHealth(ui, soh) {
  const fill = document.getElementById("batteryFillH");
  if (!fill) return;

  const pct = Math.max(0, Math.min(Number(soh) || 0, 100));
  fill.style.width = pct + "%";

  let c1, c2;
  if (pct >= 80) {
    c1 = "#12d18e"; c2 = "#0ef3a5";
    fill.classList.add("glowPulse");
  } else if (pct >= 50) {
    c1 = "#facc15"; c2 = "#fbbf24";
    fill.classList.remove("glowPulse");
  } else {
    c1 = "#ef4444"; c2 = "#f87171";
    fill.classList.remove("glowPulse");
  }

  fill.style.background = `linear-gradient(90deg, ${c1}, ${c2}, ${c1})`;
  fill.style.filter = `drop-shadow(0 0 10px ${c1}60)`;
  ui.sohText.textContent = Number.isFinite(pct) ? Math.round(pct) + "%" : "—";
}

/* [JS:CONFIDENCE] */
export function renderConfidence(ui, confidenceRaw) {
  let pct = Number(confidenceRaw);

  // Accept backend style (0..1) OR percent style (0..100)
  if (Number.isFinite(pct) && pct <= 1) pct = pct * 100;

  if (!Number.isFinite(pct)) {
    ui.confidenceCard.style.display = "none";
    return;
  }

  pct = Math.max(0, Math.min(pct, 100));

  let label = "Low";
  if (pct >= 80) label = "High";
  else if (pct >= 60) label = "Moderate";

  ui.confidenceText.textContent = `${Math.round(pct)}% — ${label}`;
  ui.confidenceCard.style.display = "block";
}