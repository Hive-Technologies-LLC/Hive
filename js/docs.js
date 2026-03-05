/* ============================================================
  HiveOS Website — Docs Drawer
  File: js/docs.js
  Rev: 0.1.0
  Purpose: Mobile docs sidebar drawer open/closed

QUICK JUMPS

[JS:DOCS-DRAWER]

============================================================ */

/* [JS:DOCS-DRAWER] */
(function(){
  const openBtn = document.querySelector("[data-docs-open]");
  const closeTargets = document.querySelectorAll("[data-docs-close]");
  const drawer = document.querySelector("[data-docs-drawer]");

  if(!openBtn || !drawer) return;

  function openDrawer(){
    document.body.classList.add("docs-drawer-open");
  }

  function closeDrawer(){
    document.body.classList.remove("docs-drawer-open");
  }

  openBtn.addEventListener("click", openDrawer);

  closeTargets.forEach(el => {
    el.addEventListener("click", closeDrawer);
  });

  // Close on Escape
  window.addEventListener("keydown", (e) => {
    if(e.key === "Escape") closeDrawer();
  });

  // Close drawer after clicking any nav link (mobile usability)
  drawer.addEventListener("click", (e) => {
    const a = e.target.closest("a");
    if(a) closeDrawer();
  });
})();