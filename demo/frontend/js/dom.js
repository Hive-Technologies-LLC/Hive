/* ============================================================
  HiveOS Lite — DOM Helpers
  File: dom.js
  Rev: 0.1.1
  Purpose:
    Single source of truth for DOM element references.
    Reduces typos + repeated getElementById calls.

  QUICK JUMPS (Search These Tags):
    [JS:EXPORTS]
    [JS:HELPERS]
    [JS:MAIN]
============================================================ */

/* [JS:HELPERS] */
export const el = (id) => document.getElementById(id);

/* [JS:EXPORTS] */
export const ui = {
  file: el("file"),
  status: el("status"),

  uploadBox: el("uploadBox"),
  results: el("results"),

  socText: el("socText"),
  sohText: el("sohText"),
  tempText: el("tempText"),

  faultCount: el("faultCount"),
  faultList: el("faultList"),

  sohCard: el("sohCard"),
  tempCard: el("tempCard"),
  faultCard: el("faultCard"),

  gauge: el("gauge"),
  spark: el("spark"),

  statVoltage: el("statVoltage"),
  statCurrent: el("statCurrent"),
  statRange: el("statRange"),
  statTTF: el("statTTF"),

  confidenceCard: el("confidenceCard"),
  confidenceText: el("confidenceText"),
};

/* [JS:MAIN]
  - Import { ui } everywhere instead of grabbing elements again.
*/