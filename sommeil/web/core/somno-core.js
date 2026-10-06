// GÉNÉRÉ par web/tools/build-core.mjs — ne pas modifier à la main.
var SomnoCore = (function () {
  'use strict';
  const __posted = [];
  var sampleRate = 48000;
  var currentTime = 0;
  class AudioWorkletProcessor { constructor() { this.port = { postMessage: (m) => __posted.push(m), onmessage: null }; } }
  let __Proc = null;
  function registerProcessor(name, cls) { __Proc = cls; }
// Primitives de traitement du signal — port exact de analysis/somno/dsp.py.
// Les tableaux multi-canaux sont des listes de Float64Array (un par canal).

const Q = { 2: [1 / Math.SQRT2], 4: [0.541196100146197, 1.3065629648763766] };

function biquad(kind, fc, fs, q) {
  const w0 = (2 * Math.PI * fc) / fs;
  const c = Math.cos(w0), s = Math.sin(w0);
  const alpha = s / (2 * q);
  let b;
  if (kind === 'lp') b = [(1 - c) / 2, 1 - c, (1 - c) / 2];
  else if (kind === 'hp') b = [(1 + c) / 2, -(1 + c), (1 + c) / 2];
  else throw new Error(kind);
  const a0 = 1 + alpha;
  return [b[0] / a0, b[1] / a0, b[2] / a0, (-2 * c) / a0, (1 - alpha) / a0];
}

function butterSos(fs, lo, hi, order = 4) {
  const sos = [];
  if (lo != null) for (const q of Q[order]) sos.push(biquad('hp', lo, fs, q));
  if (hi != null) for (const q of Q[order]) sos.push(biquad('lp', hi, fs, q));
  return sos;
}

// Forme directe II transposée, état initial nul (identique à scipy.signal.sosfilt).
function sosfiltInPlace(sos, x) {
  for (const [b0, b1, b2, a1, a2] of sos) {
    let z1 = 0, z2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      const y = b0 * xi + z1;
      z1 = b1 * xi - a1 * y + z2;
      z2 = b2 * xi - a2 * y;
      x[i] = y;
    }
  }
}

/** Filtrage à phase nulle d'un canal (Float64Array) : réflexion impaire, aller, retour. */
function filtfilt(x, fs, lo = null, hi = null, order = 4) {
  const sos = butterSos(fs, lo, hi, order);
  const fmin = Math.min(...[lo, hi].filter((f) => f != null));
  const n = x.length;
  const pad = Math.min(n - 1, Math.round((3 * fs) / fmin));
  const y = new Float64Array(n + 2 * pad);
  for (let i = 0; i < pad; i++) y[i] = 2 * x[0] - x[pad - i];
  y.set(x, pad);
  for (let i = 0; i < pad; i++) y[pad + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  sosfiltInPlace(sos, y);
  y.reverse();
  sosfiltInPlace(sos, y);
  y.reverse();
  return y.slice(pad, pad + n);
}

const filtfiltMulti = (chs, fs, lo, hi, order) => chs.map((c) => filtfilt(c, fs, lo, hi, order));

function resampleLinear(x, fsIn, fsOut) {
  const n = x.length;
  const m = Math.floor(((n - 1) * fsOut) / fsIn) + 1;
  const y = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const pos = (k / fsOut) * fsIn;
    const i0 = Math.min(Math.floor(pos), n - 2);
    const fr = pos - i0;
    y[k] = x[i0] * (1 - fr) + x[i0 + 1] * fr;
  }
  return y;
}

function median(arr) {
  const a = Float64Array.from(arr).filter((v) => Number.isFinite(v)).sort();
  if (!a.length) return NaN;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/** Percentile avec interpolation linéaire (comme numpy.percentile par défaut). */
function percentile(arr, p) {
  const a = Float64Array.from(arr).filter((v) => Number.isFinite(v)).sort();
  if (!a.length) return NaN;
  const pos = ((a.length - 1) * p) / 100;
  const i = Math.floor(pos);
  const fr = pos - i;
  return i + 1 < a.length ? a[i] * (1 - fr) + a[i + 1] * fr : a[i];
}

function mean(a) {
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

function std(a) {
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return Math.sqrt(s / a.length);
}

function robustScale(arr) {
  const a = Array.from(arr).filter((v) => Number.isFinite(v));
  if (!a.length) return 1.0;
  const med = median(a);
  const mad = median(a.map((v) => Math.abs(v - med)));
  if (mad > 0) return mad * 1.4826;
  return std(a) || 1.0;
}

// ------------------------------------------------------------------ FFT (radix 2)
function fft(re, im, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

/** Autocorrélation normalisée (biaisée) pour les décalages 0..maxLag. */
function autocorr(x, maxLag) {
  const n = x.length;
  const m = mean(x);
  let nfft = 1;
  while (nfft < 2 * n) nfft <<= 1;
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let i = 0; i < n; i++) re[i] = x[i] - m;
  fft(re, im);
  for (let i = 0; i < nfft; i++) { re[i] = re[i] * re[i] + im[i] * im[i]; im[i] = 0; }
  fft(re, im, true);
  const r = new Float64Array(maxLag + 1);
  if (re[0] > 0) for (let k = 0; k <= maxLag; k++) r[k] = re[k] / re[0];
  return r;
}

function parabolicPeak(y, i) {
  if (i > 0 && i < y.length - 1) {
    const d = y[i - 1] - 2 * y[i] + y[i + 1];
    if (d < 0) return i + (0.5 * (y[i - 1] - y[i + 1])) / d;
  }
  return i;
}

/** Périodogramme Hann + zéro-padding → puissance pour les fréquences k·fs/nfft. */
function hannSpectrum(x, nfft) {
  const n = x.length;
  const m = mean(x);
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let i = 0; i < n; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / Math.max(1, n - 1));
    re[i] = (x[i] - m) * w;
  }
  fft(re, im);
  const p = new Float64Array(nfft / 2 + 1);
  for (let k = 0; k <= nfft / 2; k++) p[k] = re[k] * re[k] + im[k] * im[k];
  return p;
}

function windowSlice(x, center, half) {
  const a = Math.max(0, center - half);
  const b = Math.min(x.length, center + half);
  return x.subarray ? x.subarray(a, b) : x.slice(a, b);
}

/** Valeurs propres/vecteurs propres d'une matrice symétrique (Jacobi). */
function eighSym(A) {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const vals = a.map((r, i) => r[i]);
  return { vals, vecs: v }; // vecteur propre i = colonne i
}

// Sonar FMCW — traitement en ligne des profils de distance envoyés par l'AudioWorklet.
// Port de analysis/somno/sonar.py (align_direct + displacement), époque par époque,
// pour ne stocker que l'onde de déplacement thoracique (≈ 6 Mo/nuit au lieu de 2 Go).

const C_SOUND = 343;
const N = 1024;
const DEC = 4;
const F0 = 18000, F1 = 20500;
const EPOCH_S = 30;
const MIN_BIN = 10;
const SMOOTH = 47;

function sonarConsts(fs) {
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

class SonarOnline {
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

  const __worklet = (function () {
// Processeur audio temps réel (AudioWorklet) :
//  1. caractéristiques sonores toutes les 250 ms (niveau, 6 bandes, crête) ;
//  2. tampon circulaire 10 s à 16 kHz pour sauver un extrait quand un bruit survient ;
//  3. sonar FMCW optionnel : émet un train de chirps 18–20,5 kHz et renvoie, pour chaque
//     période de 1024 échantillons, le profil de distance complexe (corrélation circulaire
//     avec le chirp émis, bande du chirp uniquement), décimé d'un facteur 4.

const N = 1024;
const F0 = 18000, F1 = 20500;
const BANDS = [[50, 150], [150, 500], [500, 1000], [1000, 2000], [2000, 4000], [4000, 8000]];
const AFFT = 2048;

function fft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

class SomnoProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.sonar = !!o.sonar;
    this.gain = o.sonarGain ?? 0.25;
    this.fs = sampleRate;
    // --- caractéristiques
    this.abuf = new Float32Array(AFFT);
    this.apos = 0;
    this.win = new Float32Array(AFFT).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (AFFT - 1)));
    this.frameLen = Math.round(this.fs * 0.25);
    this.fcount = 0;
    this.sumSq = 0;
    this.peak = 0;
    this.bandAcc = new Float64Array(BANDS.length);
    this.nSpec = 0;
    this.bandIdx = BANDS.map(([a, b]) => [Math.ceil((a * AFFT) / this.fs), Math.floor((b * AFFT) / this.fs)]);
    // --- extraits 16 kHz
    this.dec = Math.max(1, Math.round(this.fs / 16000));
    this.ring = new Float32Array(10 * Math.round(this.fs / this.dec));
    this.rpos = 0;
    this.dacc = 0; this.dcnt = 0;
    // --- sonar
    if (this.sonar) {
      const T = N / this.fs;
      this.chirp = new Float32Array(N);
      const m = Math.floor((0.08 * N) / 2);
      for (let i = 0; i < N; i++) {
        const t = i / this.fs;
        let v = Math.sin(2 * Math.PI * (F0 * t + (0.5 * (F1 - F0) / T) * t * t));
        if (i < m) v *= 0.5 - 0.5 * Math.cos((Math.PI * i) / m);
        if (i >= N - m) v *= 0.5 - 0.5 * Math.cos((Math.PI * (N - 1 - i)) / m);
        this.chirp[i] = v;
      }
      const re = Float64Array.from(this.chirp), im = new Float64Array(N);
      fft(re, im, false);
      this.Tre = re; this.Tim = im;
      this.mask = new Uint8Array(N);
      for (let k = 0; k < N; k++) {
        const f = k <= N / 2 ? (k * this.fs) / N : ((k - N) * this.fs) / N;
        this.mask[k] = f >= F0 - 300 && f <= F1 + 300 ? 1 : 0;
      }
      this.spos = 0;
      this.sframe = new Float64Array(N);
      this.maxBin = Math.min(N, Math.floor(((2 * 2.5) / 343) * this.fs));
      this.sDec = 4;
    }
    this.port.onmessage = (e) => {
      if (e.data?.type === 'clip') {
        const out = new Float32Array(this.ring.length);
        out.set(this.ring.subarray(this.rpos));
        out.set(this.ring.subarray(0, this.rpos), this.ring.length - this.rpos);
        this.port.postMessage({ type: 'clip', id: e.data.id, rate: this.fs / this.dec, data: out }, [out.buffer]);
      } else if (e.data?.type === 'gain') {
        this.gain = e.data.value;
      }
    };
  }

  spectrum() {
    const re = new Float64Array(AFFT), im = new Float64Array(AFFT);
    for (let i = 0; i < AFFT; i++) re[i] = this.abuf[i] * this.win[i];
    fft(re, im, false);
    this.bandIdx.forEach(([a, b], j) => {
      let s = 0;
      for (let k = a; k <= b; k++) s += re[k] * re[k] + im[k] * im[k];
      this.bandAcc[j] += s;
    });
    this.nSpec++;
  }

  sonarFrame() {
    const re = Float64Array.from(this.sframe), im = new Float64Array(N);
    fft(re, im, false);
    for (let k = 0; k < N; k++) {
      if (!this.mask[k]) { re[k] = 0; im[k] = 0; continue; }
      const a = re[k], b = im[k], c = this.Tre[k], d = -this.Tim[k];
      re[k] = a * c - b * d; im[k] = a * d + b * c;
    }
    fft(re, im, true);
    const nb = Math.floor(N / this.sDec);
    const pr = new Float32Array(nb), pi = new Float32Array(nb);
    for (let i = 0; i < nb; i++) { pr[i] = re[i * this.sDec]; pi[i] = im[i * this.sDec]; }
    this.port.postMessage({ type: 'sonar', t: currentTime, re: pr, im: pi }, [pr.buffer, pi.buffer]);
  }

  process(inputs, outputs) {
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0] && outputs[0][0];
    const n = out ? out.length : inp ? inp.length : 128;
    for (let i = 0; i < n; i++) {
      if (this.sonar && out) {
        out[i] = this.gain * this.chirp[this.spos];
        for (let c = 1; c < outputs[0].length; c++) outputs[0][c][i] = out[i];
      }
      const x = inp ? inp[i] : 0;
      // caractéristiques
      this.abuf[this.apos++] = x;
      if (this.apos === AFFT) { this.spectrum(); this.apos = 0; }
      this.sumSq += x * x;
      const ax = Math.abs(x);
      if (ax > this.peak) this.peak = ax;
      if (++this.fcount >= this.frameLen) {
        const db = (v) => 10 * Math.log10(v + 1e-20);
        const feat = [db(this.sumSq / this.fcount)];
        for (let j = 0; j < BANDS.length; j++) feat.push(db(this.bandAcc[j] / Math.max(1, this.nSpec) / AFFT));
        feat.push(20 * Math.log10(this.peak + 1e-10));
        this.port.postMessage({ type: 'feat', t: currentTime, v: feat });
        this.fcount = 0; this.sumSq = 0; this.peak = 0; this.bandAcc.fill(0); this.nSpec = 0;
      }
      // tampon d'extraits
      this.dacc += x;
      if (++this.dcnt === this.dec) {
        this.ring[this.rpos] = this.dacc / this.dec;
        this.rpos = (this.rpos + 1) % this.ring.length;
        this.dacc = 0; this.dcnt = 0;
      }
      // sonar : trames alignées sur la période d'émission
      if (this.sonar) {
        this.sframe[this.spos] = x;
        if (++this.spos === N) { this.spos = 0; this.sonarFrame(); }
      }
    }
    return true;
  }
}

registerProcessor('somno-audio', SomnoProcessor);

  })();
  let proc = null, so = null, feats = [], epochs = [], clips = [];
  return {
    init(opts) {
      sampleRate = opts.sampleRate || 48000;
      proc = new __Proc({ processorOptions: { sonar: !!opts.sonar, sonarGain: 0 } });
      so = opts.sonar ? new SonarOnline(sampleRate, (r) => epochs.push({ epoch: r.epoch, bin: r.bin, distM: r.distM, disp: Array.from(r.disp), act: Array.from(r.act) })) : null;
      feats = []; epochs = []; clips = [];
    },
    /** x : Float32Array (ou tableau) d'échantillons micro mono. */
    pushAudio(x) {
      const blk = 128;
      const out = [new Float32Array(blk)];
      for (let s = 0; s < x.length; s += blk) {
        const sub = x.subarray ? x.subarray(s, Math.min(x.length, s + blk)) : Float32Array.from(x.slice(s, s + blk));
        const o = sub.length === blk ? out : [new Float32Array(sub.length)];
        proc.process([[sub]], [o]);
        currentTime += sub.length / sampleRate;
      }
      while (__posted.length) {
        const m = __posted.shift();
        if (m.type === 'feat') feats.push(...m.v);
        else if (m.type === 'sonar' && so) so.push(m.re, m.im);
        else if (m.type === 'clip') clips.push({ t: m.id, rate: m.rate, data: Array.from(m.data) });
      }
    },
    requestClip(id) { proc.port.onmessage && proc.port.onmessage({ data: { type: 'clip', id } }); this.pushAudio(new Float32Array(0)); },
    finish() { if (so) so.finish(); },
    /** Résultats accumulés depuis le dernier appel. */
    drain() { const r = { feats, epochs, clips }; feats = []; epochs = []; clips = []; return r; },
    get time() { return currentTime; },
  };
})();
if (typeof module !== 'undefined') module.exports = SomnoCore;
