/* ============================================================
  HiveOS Lite — Charts
  File: charts.js
  Rev: 0.1.0
  Purpose:
    Canvas render functions:
      - SOC Gauge
      - SOC Sparkline

  QUICK JUMPS (Search These Tags):
    [JS:EXPORTS]
    [JS:GAUGE]
    [JS:SPARKLINE]
============================================================ */

/* [JS:GAUGE] */
export function drawGauge(canvas, percent) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const r = Math.min(w, h) / 2 - 12;

  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w / 2, h / 2);

  ctx.lineWidth = 18;
  ctx.strokeStyle = "#1b2630";
  ctx.beginPath();
  ctx.arc(0, 0, r, Math.PI * 0.75, Math.PI * 2.25);
  ctx.stroke();

  const p = Math.max(0, Math.min(percent || 0, 100)) / 100;
  const end = Math.PI * 0.75 + (Math.PI * 1.5) * p;

  const grad = ctx.createLinearGradient(-r, 0, r, 0);
  grad.addColorStop(0, "#10b981");
  grad.addColorStop(1, "#12d18e");

  ctx.strokeStyle = grad;
  ctx.shadowColor = "rgba(18,209,142,.4)";
  ctx.shadowBlur = 8;

  ctx.beginPath();
  ctx.arc(0, 0, r, Math.PI * 0.75, end);
  ctx.stroke();

  ctx.shadowBlur = 0;

  ctx.fillStyle = "#e5eef7";
  ctx.font = "900 40px ui-sans-serif, system-ui";
  const t = (percent == null || isNaN(percent)) ? "—" : Math.round(percent) + "%";
  const tw = ctx.measureText(t).width;
  ctx.fillText(t, -tw / 2, 12);

  ctx.restore();
}

/* [JS:SPARKLINE] */
export function drawSparkline(canvas, points) {
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.clientWidth);
  const h = (canvas.height = canvas.clientHeight);

  ctx.clearRect(0, 0, w, h);
  if (!points || points.length < 2) return;

  const min = Math.min(...points);
  const max = Math.max(...points);
  const y = (v) => (max === min ? h / 2 : h - ((v - min) / (max - min)) * (h - 6) - 3);

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#12d18e";

  ctx.beginPath();
  points.forEach((v, i) => {
    const x = (i / (points.length - 1)) * w;
    if (i) ctx.lineTo(x, y(v));
    else ctx.moveTo(x, y(v));
  });
  ctx.stroke();

  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(18,209,142,.25)");
  grad.addColorStop(1, "rgba(18,209,142,0)");

  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();
}