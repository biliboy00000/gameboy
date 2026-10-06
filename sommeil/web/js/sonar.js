// Sonar FMCW — traitement en ligne des profils de distance envoyés par l'AudioWorklet.
// Port de analysis/somno/sonar.py (align_direct + displacement), époque par époque,
// pour ne stocker que l'onde de déplacement thoracique (≈ 6 Mo/nuit au lieu de 2 Go).
import { fft, median } from './dsp.js';

export const C_SOUND = 343;
export const N = 1024;
export const DEC = 4;
export const F0 = 18000, F1 = 20500;
export const EPOCH_S = 30;
export const MIN_BIN = 10;
const SMOOTH = 47;

export function sonarConsts(fs) {
  const frameRate = fs / N;
  return {
    frameRate,
    ne: Math.round(EPOCH_S * frameRate),
    half: Math.floor((60 * frameRate) / 2),
    nKeep: Math.floor(((2 * 2.5) / C_SOUND) * fs / DEC),
    lam: C_SOUND / ((F0 + F1) / 2),
  };
}

function fitCircle(re, im) {
  // moindres carrés [x y 1]·[A B C] = x²+y² ; centre = (A/2, B/2)
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], v = [0, 0, 0];
  for (let i = 0; i < re.length; i++) {
    const r = [re[i], im[i], 1], b = re[i] * re[i] + im[i] * im[i];
    for (let p = 0; p < 3; p++) { v[p] += r[p] * b; for (let q = 0; q < 3; q++) M[p][q] += r[p] * r[q]; }
  }
  const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = det(M);
  if (!Number.isFinite(D) || Math.abs(D) < 1e-300) return null;
  const sol = [0, 1, 2].map((k) => det(M.map((row, p) => row.map((x, q) => (q === k ? v[p] : x)))) / D);
  return [sol[0] / 2, sol[1] / 2];
}

function unwrap(a) {
  const out = Float64Array.from(a);
  let off = 0;
  for (let i = 1; i < a.length; i++) {
    let d = a[i] - a[i - 1];
    if (d > Math.PI) off -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
    else if (d < -Math.PI) off -= 2 * Math.PI * Math.round(d / (2 * Math.PI));
    out[i] = a[i] + off;
  }
  return out;
}

export class SonarOnline {
  /** onEpoch({epoch, disp: Float32Array(ne), act: Float32Array(ne), bin}) */
  constructor(fs, onEpoch) {
    Object.assign(this, sonarConsts(fs));
    this.fs = fs;
    this.onEpoch = onEpoch;
    this.re = []; this.im = []; this.peaks = [];
    this.base = 0;      // index absolu de re[0]
    this.count = 0;     // trames reçues
    this.nextEpoch = 0;
  }

  push(re, im) {
    let best = 0, bv = -1;
    for (let i = 0; i < re.length; i++) { const p = re[i] * re[i] + im[i] * im[i]; if (p > bv) { bv = p; best = i; } }
    this.re.push(re); this.im.push(im); this.peaks.push(best);
    this.count++;
    const h = SMOOTH >> 1;
    for (;;) {
      const c = this.nextEpoch * this.ne + (this.ne >> 1);
      if (this.count < c + this.half + h + 1) break;
      this.process(this.nextEpoch, false);
    }
  }

  finish() {
    const nEp = Math.floor(this.count / this.ne);
    while (this.nextEpoch < nEp) this.process(this.nextEpoch, true);
  }

  shiftAt(i) {
    const h = SMOOTH >> 1;
    const lo = Math.max(0, i - h), hi = Math.min(this.count, i + h + 1);
    return Math.trunc(median(this.peaks.slice(lo - this.base, hi - this.base)));
  }

  aligned(i) {
    const s = this.shiftAt(i), r = this.re[i - this.base], m = this.im[i - this.base], nb = r.length;
    const ar = new Float64Array(this.nKeep), ai = new Float64Array(this.nKeep);
    for (let k = 0; k < this.nKeep; k++) { ar[k] = r[(s + k) % nb]; ai[k] = m[(s + k) % nb]; }
    return [ar, ai];
  }

  process(e, atEnd) {
    const { ne, half, nKeep } = this;
    const c = e * ne + (ne >> 1);
    const a = Math.max(0, c - half), b = Math.min(this.count, c + half);
    const L = b - a;
    const rows = [];
    for (let i = a; i < b; i++) rows.push(this.aligned(i));
    let nfft = 1; while (nfft < Math.max(2, L)) nfft <<= 1;
    const df = this.frameRate / nfft;
    let bestJ = 0, bestS = -Infinity;
    for (let j = MIN_BIN; j < nKeep; j++) {
      let mr = 0, mi = 0;
      for (const [r, m] of rows) { mr += r[j]; mi += m[j]; }
      mr /= L; mi /= L;
      const xr = new Float64Array(nfft), xi = new Float64Array(nfft);
      for (let t = 0; t < L; t++) { xr[t] = rows[t][0][j] - mr; xi[t] = rows[t][1][j] - mi; }
      fft(xr, xi);
      let pb = 0, ph = 0;
      for (let k = 0; k < nfft; k++) {
        const f = Math.abs((k <= nfft / 2 ? k : k - nfft) * df);
        const p = xr[k] * xr[k] + xi[k] * xi[k];
        if (f >= 0.1 && f <= 0.7) pb += p;
        else if (f > 0.7 && f <= 5.0) ph += p;
      }
      ph += 1e-30;
      const score = (pb * pb) / (pb + ph);
      if (score > bestS) { bestS = score; bestJ = j; }
    }
    const zr = rows.map((r) => r[0][bestJ]), zi = rows.map((r) => r[1][bestJ]);
    let mr = 0, mi = 0;
    for (let t = 0; t < L; t++) { mr += zr[t]; mi += zi[t]; }
    mr /= L; mi /= L;
    let cen = fitCircle(zr, zi);
    const mags = zr.map((v, t) => Math.hypot(v - mr, zi[t] - mi));
    const mm = mags.reduce((s, v) => s + v, 0) / L;
    const sd = Math.sqrt(mags.reduce((s, v) => s + (v - mm) * (v - mm), 0) / L);
    if (!cen || !Number.isFinite(cen[0]) || Math.hypot(cen[0] - mr, cen[1] - mi) > 50 * sd + 1e-12) cen = [mr, mi];
    const ang = unwrap(zr.map((v, t) => Math.atan2(zi[t] - cen[1], v - cen[0])));
    const lo = e * ne - a;
    const disp = new Float32Array(ne);
    let pm = 0;
    for (let t = 0; t < ne; t++) { disp[t] = (ang[lo + t] * this.lam) / (4 * Math.PI) * 1000; pm += disp[t]; }
    pm /= ne;
    for (let t = 0; t < ne; t++) disp[t] -= pm;
    const act = new Float32Array(ne);
    for (let t = 0; t < ne; t++) {
      const i = e * ne + t;
      if (i === 0) continue;
      const [r1, m1] = this.aligned(i), [r0, m0] = this.aligned(i - 1);
      let s = 0;
      for (let k = MIN_BIN; k < nKeep; k++) { const dr = r1[k] - r0[k], dm = m1[k] - m0[k]; s += dr * dr + dm * dm; }
      act[t] = Math.sqrt(s);
    }
    this.onEpoch({ epoch: e, disp, act, bin: bestJ, distM: (bestJ * DEC * C_SOUND) / (2 * this.fs) });
    this.nextEpoch = e + 1;
    // libère les trames qui ne serviront plus
    const keepFrom = Math.max(0, (e + 1) * ne + (ne >> 1) - half - (SMOOTH >> 1) - 2);
    if (!atEnd && keepFrom > this.base) {
      const k = keepFrom - this.base;
      this.re.splice(0, k); this.im.splice(0, k); this.peaks.splice(0, k);
      this.base = keepFrom;
    }
  }
}
