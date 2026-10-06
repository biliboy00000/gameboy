// Événements respiratoires (chutes d'amplitude ≥ seuil pendant 10–120 s) — port de
// analysis/somno/apnea.py. Indicatif : ne remplace pas une polygraphie.
import { percentile, median } from './dsp.js';

export const FS_R = 10;
export const DROP = 0.5;

export function envelope(w, fs = FS_R, winS = 6) {
  const sec = fs;
  const ns = Math.floor(w.length / sec);
  const mx = new Float64Array(ns), mn = new Float64Array(ns);
  for (let s = 0; s < ns; s++) {
    let a = -Infinity, b = Infinity;
    for (let i = s * sec; i < (s + 1) * sec; i++) { a = Math.max(a, w[i]); b = Math.min(b, w[i]); }
    mx[s] = a; mn[s] = b;
  }
  const env = new Float64Array(ns);
  const k = winS;
  for (let i = 0; i < ns; i++) {
    const lo = Math.max(0, i - (k >> 1)), hi = Math.min(ns, i + (k >> 1) + 1);
    let a = -Infinity, b = Infinity;
    for (let j = lo; j < hi; j++) { a = Math.max(a, mx[j]); b = Math.min(b, mn[j]); }
    env[i] = a - b;
  }
  return env;
}

export function detectEvents(w, movingS = null, { fs = FS_R, drop = DROP, minS = 10, maxS = 120, baseWinS = 180 } = {}) {
  const env = envelope(w, fs);
  const ns = env.length;
  const h = baseWinS >> 1;
  const pts = [];
  for (let i = 0; i < ns; i += 10) pts.push(i);
  const bp = pts.map((i) => percentile(env.subarray(Math.max(0, i - h), i + h + 1), 75));
  const base = new Float64Array(ns);
  for (let i = 0; i < ns; i++) {
    const k = Math.min(pts.length - 1, Math.floor(i / 10));
    if (k + 1 < pts.length) {
      const f = (i - pts[k]) / (pts[k + 1] - pts[k]);
      base[i] = bp[k] * (1 - f) + bp[k + 1] * f;
    } else base[i] = bp[k];
  }
  const low = new Uint8Array(ns);
  for (let i = 0; i < ns; i++) low[i] = env[i] < (1 - drop) * base[i] ? 1 : 0;
  if (movingS) {
    const mv = new Uint8Array(ns);
    for (let i = 0; i < Math.min(ns, movingS.length); i++) mv[i] = movingS[i] ? 1 : 0;
    for (let i = 0; i < ns; i++) {
      let any = 0;
      for (let j = Math.max(0, i - 10); j <= Math.min(ns - 1, i + 10); j++) any |= mv[j];
      if (any) low[i] = 0;
    }
  }
  const events = [];
  let i = 0;
  while (i < ns) {
    if (low[i]) {
      let j = i;
      while (j < ns && low[j]) j++;
      const dur = j - i + 6;
      if (dur >= minS && dur <= maxS) {
        let mnv = Infinity;
        for (let k = i; k < j; k++) mnv = Math.min(mnv, env[k]);
        const depth = 1 - mnv / Math.max(median(base.subarray(i, j)), 1e-12);
        events.push([i - 3, j + 3, depth]);
      }
      i = j;
    } else i++;
  }
  return events;
}

export function rdi(events, sleepMaskEpochs, epochS = 30) {
  const sleepN = sleepMaskEpochs.filter(Boolean).length;
  const tstH = (sleepN * epochS) / 3600;
  if (tstH <= 0) return { rdi: 0, n: 0 };
  let n = 0;
  for (const [s] of events) {
    const k = Math.floor(Math.max(0, s) / epochS);
    if (k < sleepMaskEpochs.length && sleepMaskEpochs[k]) n++;
  }
  return { rdi: n / tstH, n };
}
