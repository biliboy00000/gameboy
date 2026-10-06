// Graphiques canvas (hypnogramme, courbes, mini-hypnogrammes) — sans bibliothèque.

const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
export const STAGE_COLORS = () => [css('--wake'), css('--light'), css('--deep'), css('--rem')];
// ordre vertical façon Apple Santé : Éveil, REM, Léger (cœur), Profond
const ROW = [0, 2, 3, 1];

function setup(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(r.width * dpr));
  canvas.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}

const hhmm = (t) => new Date(t).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });

function timeAxis(ctx, x0, x1, y, start, durMs) {
  ctx.fillStyle = css('--muted');
  ctx.font = '11px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  const hour = 3600e3;
  const first = Math.ceil(start / hour) * hour;
  const step = durMs > 8 * hour ? 2 * hour : hour;
  for (let t = first; t <= start + durMs; t += step) {
    const x = x0 + ((t - start) / durMs) * (x1 - x0);
    ctx.fillText(hhmm(t), x, y);
  }
}

export function hypnogram(canvas, hyp, start, { proba = null } = {}) {
  const { ctx, w, h } = setup(canvas);
  const left = 54, right = w - 6, top = 6, bottom = h - 22;
  const rowH = (bottom - top) / 4;
  const labels = ['Éveil', 'REM', 'Léger', 'Profond'];
  const colors = STAGE_COLORS();
  const rowColors = [colors[0], colors[3], colors[1], colors[2]];
  ctx.font = '12px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (let r = 0; r < 4; r++) {
    ctx.fillStyle = css('--muted');
    ctx.fillText(labels[r], 0, top + rowH * (r + 0.5));
    ctx.strokeStyle = css('--line');
    ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.moveTo(left, top + rowH * (r + 0.5)); ctx.lineTo(right, top + rowH * (r + 0.5)); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  const n = hyp.length;
  const dx = (right - left) / n;
  // blocs colorés + liaisons verticales
  let prevRow = null;
  for (let i = 0; i < n; i++) {
    const r = ROW[hyp[i]];
    const conf = proba ? Math.max(...proba[i]) : 1;
    ctx.fillStyle = rowColors[r];
    ctx.globalAlpha = 0.55 + 0.45 * conf;
    const y = top + rowH * r + rowH * 0.18;
    ctx.fillRect(left + i * dx, y, Math.max(dx, 0.6), rowH * 0.64);
    if (prevRow !== null && prevRow !== r) {
      ctx.globalAlpha = 0.35;
      ctx.fillRect(left + i * dx - 0.5, top + rowH * (Math.min(prevRow, r) + 0.5), 1, rowH * Math.abs(prevRow - r));
    }
    prevRow = r;
  }
  ctx.globalAlpha = 1;
  ctx.textBaseline = 'alphabetic';
  timeAxis(ctx, left, right, h - 4, start, n * 30e3);
}

export function series(canvas, values, start, { color = css('--primary'), unit = '', min = null, max = null, bars = false } = {}) {
  const { ctx, w, h } = setup(canvas);
  const left = 34, right = w - 6, top = 8, bottom = h - 22;
  const fin = values.filter((v) => v != null && Number.isFinite(v));
  if (!fin.length) {
    ctx.fillStyle = css('--muted');
    ctx.font = '13px -apple-system, sans-serif';
    ctx.fillText('Pas de mesure fiable', left, h / 2);
    return;
  }
  const sorted = fin.slice().sort((a, b) => a - b);
  let lo = min ?? sorted[Math.floor(sorted.length * 0.02)];
  let hi = max ?? sorted[Math.floor(sorted.length * 0.98)];
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.1;
  if (min == null) lo -= pad;
  if (max == null) hi += pad;
  const n = values.length;
  const X = (i) => left + ((i + 0.5) / n) * (right - left);
  const Y = (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
  ctx.fillStyle = css('--muted');
  ctx.font = '11px -apple-system, sans-serif';
  ctx.textAlign = 'right';
  [lo + pad * (min == null), hi - pad * (max == null)].forEach((v) => ctx.fillText(Math.round(v) + unit, left - 4, Y(v) + 4));
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  if (bars) {
    const bw = Math.max(1, (right - left) / n);
    values.forEach((v, i) => { if (v) ctx.fillRect(X(i) - bw / 2, Y(v), bw, bottom - Y(v)); });
  } else {
    // médiane glissante (5 époques) pour lisibilité, points bruts en filigrane
    ctx.globalAlpha = 0.25;
    values.forEach((v, i) => { if (v != null && Number.isFinite(v)) ctx.fillRect(X(i) - 0.75, Y(v) - 0.75, 1.5, 1.5); });
    ctx.globalAlpha = 1;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const win = values.slice(Math.max(0, i - 2), i + 3).filter((v) => v != null && Number.isFinite(v)).sort((a, b) => a - b);
      if (win.length < 2) { pen = false; continue; }
      const v = win[win.length >> 1];
      if (!pen) { ctx.moveTo(X(i), Y(v)); pen = true; } else ctx.lineTo(X(i), Y(v));
    }
    ctx.stroke();
  }
  timeAxis(ctx, left, right, h - 4, start, n * 30e3);
}

export function miniHyp(canvas, hyp) {
  const { ctx, w, h } = setup(canvas);
  const colors = STAGE_COLORS();
  const rowColors = [colors[0], colors[3], colors[1], colors[2]];
  const dx = w / hyp.length;
  hyp.forEach((s, i) => {
    const r = ROW[s];
    ctx.fillStyle = rowColors[r];
    ctx.fillRect(i * dx, (h / 4) * r, Math.max(dx, 0.6), h / 4);
  });
}

export function trend(canvas, nights) {
  const { ctx, w, h } = setup(canvas);
  if (!nights.length) return;
  const left = 30, right = w - 6, top = 10, bottom = h - 20;
  const maxH = Math.max(9, ...nights.map((n) => n.metrics.tstMin / 60));
  const bw = Math.min(26, (right - left) / nights.length - 4);
  const colors = STAGE_COLORS();
  nights.forEach((n, i) => {
    const x = left + (i + 0.5) * ((right - left) / nights.length) - bw / 2;
    let y = bottom;
    [2, 1, 3].forEach((s) => {
      const hh = ((n.metrics.stageMin[s] / 60) / maxH) * (bottom - top);
      ctx.fillStyle = colors[s];
      ctx.fillRect(x, y - hh, bw, hh);
      y -= hh;
    });
    ctx.fillStyle = css('--muted');
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(new Date(n.start).toLocaleDateString('fr-FR', { day: 'numeric', month: 'numeric' }), x + bw / 2, h - 5);
  });
  ctx.textAlign = 'right';
  ctx.fillStyle = css('--muted');
  for (let k = 0; k <= maxH; k += 3) {
    const y = bottom - (k / maxH) * (bottom - top);
    ctx.fillText(k + ' h', left - 4, y + 3);
  }
}

export function wave(canvas, y) {
  const { ctx, w, h } = setup(canvas);
  if (!y || y.length < 2) return;
  let lo = Infinity, hi = -Infinity;
  for (const v of y) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (hi - lo < 1e-12) hi = lo + 1;
  ctx.strokeStyle = css('--primary');
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < y.length; i++) {
    const X = (i / (y.length - 1)) * w, Y = h - 6 - ((y[i] - lo) / (hi - lo)) * (h - 12);
    i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
  }
  ctx.stroke();
}
