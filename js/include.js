/* ============================================================
  HiveOS Website — Partials Loader
  File: js/include.js
  Rev: 0.1.1
  Purpose: Loads shared HTML partials into pages using data-include

  QUICK JUMPS (Search These Tags):
    [JS:CONFIG]
    [JS:HELPERS]
    [JS:INCLUDE-ENGINE]
    [JS:BOOT]
============================================================ */


/* ============================================================
   [JS:CONFIG]
============================================================ */
const INCLUDE_SELECTOR = "[data-include]";
const FETCH_OPTIONS = { cache: "no-store" }; // prevents stale partials while developing


/* ============================================================
   [JS:HELPERS]
============================================================ */
function setMissingPartial(node, file){
  node.innerHTML =
    `<div class="card card--padded" style="max-width:1120px;margin:16px auto;">
      <div class="card__title">Missing Partial</div>
      <p class="card__muted">Could not load: <code>${file}</code></p>
    </div>`;
}

async function fetchText(file){
  const res = await fetch(file, FETCH_OPTIONS);
  if (!res.ok) throw new Error(`Failed to load ${file} (${res.status})`);
  return await res.text();
}


/* ============================================================
   [JS:INCLUDE-ENGINE]
============================================================ */
async function includePartials(){
  const nodes = document.querySelectorAll(INCLUDE_SELECTOR);

  for (const node of nodes){
    const file = node.getAttribute("data-include");
    if (!file){
      continue;
    }

    try{
      node.innerHTML = await fetchText(file);
    }catch(err){
      console.error(err);
      setMissingPartial(node, file);
    }
  }
}


/* ============================================================
   [JS:BOOT]
============================================================ */
document.addEventListener("DOMContentLoaded", includePartials);