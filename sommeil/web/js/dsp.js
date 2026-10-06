// Primitives de traitement du signal — port exact de analysis/somno/dsp.py.
// Les tableaux multi-canaux sont des listes de Float64Array (un par canal).

const Q = { 2: [1 / Math.SQRT2], 4: [0.541196100146197, 1.3065629648763766] };

export function biquad(kind, fc, fs, q) {
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

export function butterSos(fs, lo, hi, order = 4) {
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
export function filtfilt(x, fs, lo = null, hi = null, order = 4) {
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

export const filtfiltMulti = (chs, fs, lo, hi, order) => chs.map((c) => filtfilt(c, fs, lo, hi, order));

export function resampleLinear(x, fsIn, fsOut) {
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

export function median(arr) {
  const a = Float64Array.from(arr).filter((v) => Number.isFinite(v)).sort();
  if (!a.length) return NaN;
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/** Percentile avec interpolation linéaire (comme numpy.percentile par défaut). */
export function percentile(arr, p) {
  const a = Float64Array.from(arr).filter((v) => Number.isFinite(v)).sort();
  if (!a.length) return NaN;
  const pos = ((a.length - 1) * p) / 100;
  const i = Math.floor(pos);
  const fr = pos - i;
  return i + 1 < a.length ? a[i] * (1 - fr) + a[i + 1] * fr : a[i];
}

export function mean(a) {
  let s = 0;
  for (const v of a) s += v;
  return s / a.length;
}

export function std(a) {
  const m = mean(a);
  let s = 0;
  for (const v of a) s += (v - m) * (v - m);
  return Math.sqrt(s / a.length);
}

export function robustScale(arr) {
  const a = Array.from(arr).filter((v) => Number.isFinite(v));
  if (!a.length) return 1.0;
  const med = median(a);
  const mad = median(a.map((v) => Math.abs(v - med)));
  if (mad > 0) return mad * 1.4826;
  return std(a) || 1.0;
}

// ------------------------------------------------------------------ FFT (radix 2)
export function fft(re, im, inverse = false) {
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
export function autocorr(x, maxLag) {
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

export function parabolicPeak(y, i) {
  if (i > 0 && i < y.length - 1) {
    const d = y[i - 1] - 2 * y[i] + y[i + 1];
    if (d < 0) return i + (0.5 * (y[i - 1] - y[i + 1])) / d;
  }
  return i;
}

/** Périodogramme Hann + zéro-padding → puissance pour les fréquences k·fs/nfft. */
export function hannSpectrum(x, nfft) {
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

export function windowSlice(x, center, half) {
  const a = Math.max(0, center - half);
  const b = Math.min(x.length, center + half);
  return x.subarray ? x.subarray(a, b) : x.slice(a, b);
}

/** Valeurs propres/vecteurs propres d'une matrice symétrique (Jacobi). */
export function eighSym(A) {
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
