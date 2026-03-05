/* ============================================================
  HiveOS Lite — Main Orchestrator
  File: main.js
  Rev: 0.1.1
  Purpose:
    App entry point:
      - file selection
      - API request
      - renders results into UI

  QUICK JUMPS (Search These Tags):
    [JS:IMPORTS]
    [JS:MAIN]
    [JS:BOOTSTRAP]
    [JS:ERRORS]
============================================================ */

/* [JS:IMPORTS] */
import { ui } from "./dom.js";
import { isValidFile } from "./validators.js";
import { computeAvgVIFromFile } from "./file_parse.js";
import { analyzeFileWithAPI } from "./api.js";
import { drawGauge, drawSparkline } from "./charts.js";
import {
  populateSocStats,
  renderFaults,
  renderBatteryHealth,
  renderConfidence
} from "./render.js";

/* [JS:MAIN] */
async function analyzeSelectedFile() {
  if (!ui.file.files.length) return;

  const f = ui.file.files[0];

  if (!isValidFile(f)) {
    ui.status.textContent = "Invalid file type. Please upload .csv or .json.";
    return;
  }

  // Optional: clear/hide cards before new render (prevents stale UI)
  ui.status.textContent = "Analyzing…";
  ui.sohCard.style.display = "none";
  ui.tempCard.style.display = "none";
  ui.faultCard.style.display = "none";
  ui.confidenceCard.style.display = "none";

  try {
    const res = await analyzeFileWithAPI(f);

    if (!res.ok) {
      if (res.status === 413) ui.status.textContent = "Error: File too large (limit ~10MB).";
      else if (res.status === 429) ui.status.textContent = "Rate limited: please wait a bit.";
      else ui.status.textContent = "Server error: " + res.status;
      return;
    }

    const data = await res.json();

    ui.uploadBox.style.display = "none";
    ui.results.classList.add("show");

    const soc = data["SOC (%)"];
    const soh = data["SOH (%)"];
    const flags = data["Flagged Cells"] || [];
    const stats = data?.stats || {};

    // ✅ Correct confidence source (backend provides 0..1)
    const confidenceSoc = data?.methods?.soc?.confidence; // preferred
    // const confidenceSoh = data?.methods?.soh?.confidence; // optional

    const { vAvg: vAvgFile, iAvg: iAvgFile } = await computeAvgVIFromFile(f);

    drawGauge(ui.gauge, soc);
    ui.socText.textContent = Number.isFinite(soc) ? Math.round(soc) + "%" : "—";

    const sparkPts = Array.from(
      { length: 24 },
      (_, i) => (soc || 0) + Math.sin(i / 3) * 4 + (Math.random() - 0.5) * 2
    );
    drawSparkline(ui.spark, sparkPts);

    populateSocStats(ui, data, soc, vAvgFile, iAvgFile);

    // ✅ Confidence card render (renderConfidence accepts 0..1 or 0..100)
    renderConfidence(ui, confidenceSoc);

    if (Number.isFinite(soh)) {
      renderBatteryHealth(ui, soh);
      ui.sohCard.style.display = "block";
    }

    if (Number.isFinite(stats.temp_avg_c)) {
      ui.tempText.textContent = Math.round(stats.temp_avg_c) + "°C";
      ui.tempCard.style.display = "block";
    }

    if (flags.length > 0) {
      ui.faultCount.textContent = flags.length;
      renderFaults(ui, flags);
      ui.faultCard.style.display = "block";
    } else {
      renderFaults(ui, []);
    }

    ui.status.textContent = "Done.";
  } catch (e) {
    /* [JS:ERRORS] */
    ui.results.classList.remove("show");
    ui.results.style.display = "none";
    ui.uploadBox.style.display = "block";
    ui.status.textContent = "Error: " + (e?.message || e);
  }
}

/* [JS:BOOTSTRAP] */
ui.file.addEventListener("change", analyzeSelectedFile);
drawGauge(ui.gauge, 0);
drawSparkline(ui.spark, [0, 1, 0.6, 0.8, 0.7]);