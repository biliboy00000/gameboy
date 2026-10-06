// Analyse d'une nuit — port exact de analysis/somno/{respiration,heart,movement,features,pipeline}.py
// Vérifié numériquement contre Python par analysis/tests/test_parity.py.
import {
  filtfilt, resampleLinear, median, percentile, mean, std, robustScale, autocorr,
  parabolicPeak, hannSpectrum, eighSym,
} from './dsp.js';

export const EPOCH_S = 30;
export const FS_R = 10;
const G = 9.80665;
const BAND = [0.08, 1.0];

// ============================================================ respiration
export function motionToRespChannels(acc, gyro, fs) {
  const ang = gyro.map((g) => {
    const f = filtfilt(g, fs, 0.03, null, 2);
    const out = new Float64Array(f.length);
    let s = 0;
    for (let i = 0; i < f.length; i++) { s += f[i]; out[i] = G * (s / fs); }
    return out;
  });
  return [...acc, ...ang].map((x) => {
    let y = filtfilt(x, fs, null, 2.0, 4);
    y = resampleLinear(y, fs, FS_R);
    y = filtfilt(y, FS_R, BAND[0], BAND[1], 2);
    // blanchiment : divise par le niveau de bruit (échelle robuste de la dérivée)
    const d = new Float64Array(y.length - 1);
    for (let i = 1; i < y.length; i++) d[i - 1] = y[i] - y[i - 1];
    const sc = robustScale(d);
    return y.map((v) => v / (sc > 0 ? sc : 1));
  });
}

export function respChannelsFromSignal(x, fs) {
  let y = filtfilt(x, fs, null, 2.0, 4);
  y = resampleLinear(y, fs, FS_R);
  return [filtfilt(y, FS_R, BAND[0], BAND[1], 2)];
}

export function pcaWaveform(chans, winS = 120) {
  const c = chans.length;
  const n = chans[0].length;
  const ne = EPOCH_S * FS_R;
  const nEp = Math.floor(n / ne);
  const half = Math.floor((winS * FS_R) / 2);
  const out = new Float64Array(nEp * ne);
  let prev = null;
  // énergie par seconde : secondes agitées exclues de l'ACP
  const sec = FS_R;
  const nS = Math.floor(n / sec);
  const eS = new Float64Array(nS);
  for (let s = 0; s < nS; s++) {
    let acc = 0;
    for (let k = s * sec; k < (s + 1) * sec; k++) for (let i = 0; i < c; i++) acc += chans[i][k] * chans[i][k];
    eS[s] = acc / sec;
  }
  for (let e = 0; e < nEp; e++) {
    const ctr = e * ne + Math.floor(ne / 2);
    const a = Math.max(0, ctr - half), b = Math.min(n, ctr + half);
    const sa = Math.floor(a / sec), sb = Math.min(nS, Math.floor(b / sec));
    const es = eS.slice(sa, sb);
    const lim = 4 * median(es);
    let rows = [];
    for (let s = sa; s < sb; s++) if (eS[s] <= lim) for (let k = s * sec; k < (s + 1) * sec; k++) rows.push(k);
    if (rows.length < 10 * sec) { rows = []; for (let k = a; k < b; k++) rows.push(k); }
    const cov = Array.from({ length: c }, () => new Array(c).fill(0));
    for (let i = 0; i < c; i++) {
      for (let j = i; j < c; j++) {
        let s = 0;
        const xi = chans[i], xj = chans[j];
        for (const k of rows) s += xi[k] * xj[k];
        cov[i][j] = cov[j][i] = s / Math.max(1, rows.length);
      }
    }
    const { vals, vecs } = eighSym(cov);
    let best = 0;
    for (let i = 1; i < c; i++) if (vals[i] > vals[best]) best = i;
    let w = vecs.map((row) => row[best]);
    if (prev) {
      if (w.reduce((s, v, i) => s + v * prev[i], 0) < 0) w = w.map((v) => -v);
    } else {
      let im = 0;
      for (let i = 1; i < c; i++) if (Math.abs(w[i]) > Math.abs(w[im])) im = i;
      if (w[im] < 0) w = w.map((v) => -v);
    }
    prev = w;
    for (let k = e * ne; k < (e + 1) * ne; k++) {
      let s = 0;
      for (let i = 0; i < c; i++) s += chans[i][k] * w[i];
      out[k] = s;
    }
  }
  return out;
}

export function detectBreaths(w, fs = FS_R, winS = 60) {
  const n = w.length;
  const blk = 10 * fs;
  const nb = Math.max(1, Math.floor(n / blk));
  const loc = new Float64Array(nb);
  for (let i = 0; i < nb; i++) {
    const seg = w.slice(i * blk, (i + 1) * blk).map(Math.abs);
    loc[i] = percentile(seg, 90);
  }
  const k = Math.max(1, Math.floor(winS / 10));
  const locS = new Float64Array(nb);
  for (let i = 0; i < nb; i++) locS[i] = median(loc.slice(Math.max(0, i - (k >> 1)), i + (k >> 1) + 1));
  const h = new Float64Array(n);
  for (let i = 0; i < n; i++) h[i] = 0.15 * locS[Math.min(nb - 1, Math.floor(i / blk))];
  const times = [];
  let state = 0, lastZc = null;
  for (let i = 1; i < n; i++) {
    const x = w[i];
    if (w[i - 1] < 0 && x >= 0) lastZc = i - 1 + -w[i - 1] / (x - w[i - 1]);
    if (x < -h[i]) state = -1;
    else if (x > h[i]) {
      if (state === -1 && lastZc !== null) {
        const t = lastZc / fs;
        if (!times.length || t - times[times.length - 1] >= 1.0) times.push(t);
      }
      state = 1;
    }
  }
  const amps = new Float64Array(times.length).fill(NaN);
  const idx = times.map((t) => Math.round(t * fs));
  for (let q = 0; q < idx.length - 1; q++) {
    let mn = Infinity, mx = -Infinity;
    for (let i = idx[q]; i <= Math.min(n - 1, idx[q + 1]); i++) { mn = Math.min(mn, w[i]); mx = Math.max(mx, w[i]); }
    amps[q] = mx - mn;
  }
  return { times: Float64Array.from(times), amps };
}

export const RESP_NAMES = ['rr', 'rr_spec', 'ibi_cv', 'amp_cv', 'reg', 'spec_conc', 'amp', 'n_breaths'];

export function respEpochFeatures(w, bt, ba, fs = FS_R, ctxS = 60) {
  const ne = EPOCH_S * fs;
  const nEp = Math.floor(w.length / ne);
  const half = Math.floor((ctxS * fs) / 2);
  const nfft = 1024;
  const F = Array.from({ length: nEp }, () => new Array(8).fill(NaN));
  for (let e = 0; e < nEp; e++) {
    const c = e * ne + Math.floor(ne / 2);
    const seg = w.slice(Math.max(0, c - half), Math.min(w.length, c + half));
    const t0 = (c - half) / fs, t1 = (c + half) / fs;
    const p = hannSpectrum(seg, nfft);
    let tot = 0, best = -1, bestV = -1;
    for (let k = 0; k < p.length; k++) {
      const f = (k * fs) / nfft;
      if (f >= BAND[0] && f <= 0.7) {
        tot += p[k];
        if (p[k] > bestV) { bestV = p[k]; best = k; }
      }
    }
    if (tot > 0) {
      const fi = (parabolicPeak(p, best) * fs) / nfft;
      F[e][1] = 60 * fi;
      const fb = (best * fs) / nfft;
      let near = 0;
      for (let k = 0; k < p.length; k++) {
        const f = (k * fs) / nfft;
        if (f >= BAND[0] && f <= 0.7 && Math.abs(f - fb) <= 0.04) near += p[k];
      }
      F[e][5] = near / tot;
      const lag = Math.round(fs / Math.max(fi, 1e-3));
      if (lag < seg.length - 1) F[e][4] = autocorr(seg, lag)[lag];
    }
    F[e][6] = std(w.subarray(e * ne, (e + 1) * ne));
    const sel = [];
    for (let i = 0; i < bt.length; i++) if (bt[i] >= t0 && bt[i] < t1) sel.push(i);
    F[e][7] = sel.length;
    if (sel.length >= 3) {
      const ibi = [];
      for (let i = 1; i < sel.length; i++) ibi.push(bt[sel[i]] - bt[sel[i - 1]]);
      F[e][0] = 60 / median(ibi);
      F[e][2] = std(ibi) / mean(ibi);
      const a = sel.map((i) => ba[i]).filter(Number.isFinite);
      if (a.length >= 3 && mean(a) > 0) F[e][3] = std(a) / mean(a);
    }
  }
  return F;
}

export function analyzeResp(chans) {
  const w = pcaWaveform(chans);
  const { times, amps } = detectBreaths(w);
  return { w, times, amps, F: respEpochFeatures(w, times, amps) };
}

// ============================================================ cœur (BCG)
export const FS_ENV = 25, SUB_S = 10, STEP_S = 5, HR_MIN = 38, HR_MAX = 150;

export function bcgEnvelopes(motion, fs, lo = 4.0, hi = 12.0) {
  return motion.map((m) => {
    const x = filtfilt(m, fs, lo, Math.min(hi, 0.45 * fs), 2);
    const sq = x.map((v) => v * v);
    const e = resampleLinear(filtfilt(sq, fs, null, 2.5, 2), fs, FS_ENV);
    const med = median(e.map(Math.abs));
    const d = med > 0 ? med : 1;
    return e.map((v) => v / d);
  });
}

export function windowedAcf(env, fs = FS_ENV) {
  const n = env[0].length;
  const lw = SUB_S * fs, ls = STEP_S * fs;
  const lagMin = Math.floor((fs * 60) / HR_MAX), lagMax = Math.ceil((fs * 60) / HR_MIN);
  const starts = [];
  for (let s = 0; s < Math.max(1, n - lw + 1); s += ls) starts.push(s);
  const acf = starts.map((s) => {
    const acc = new Float64Array(lagMax + 1);
    let wsum = 0;
    for (const ch of env) {
      const seg = ch.subarray(s, s + lw);
      if (seg.length <= lagMax + 1 || std(seg) === 0) continue;
      const r = autocorr(seg, lagMax);
      let q = 0;
      for (let k = lagMin; k <= lagMax; k++) q = Math.max(q, r[k]);
      for (let k = 0; k <= lagMax; k++) acc[k] += q * q * r[k];
      wsum += q * q;
    }
    if (wsum > 0) for (let k = 0; k <= lagMax; k++) acc[k] /= wsum;
    return acc;
  });
  const centers = starts.map((s) => (s + lw / 2) / fs);
  return { acf, centers, lagMin };
}

export function trackHr(acf, lagMin, fs = FS_ENV, penalty = 0.06) {
  const nw = acf.length;
  if (!nw) return { hr: [], q: [] };
  const nl = acf[0].length - lagMin;
  let D = Array.from(acf[0].subarray(lagMin));
  const back = [];
  for (let t = 1; t < nw; t++) {
    const bt = new Int32Array(nl);
    const nd = new Array(nl);
    for (let j = 0; j < nl; j++) {
      let bi = 0, bv = -Infinity;
      for (let i = 0; i < nl; i++) {
        const v = D[i] - penalty * Math.abs(i - j);
        if (v > bv) { bv = v; bi = i; }
      }
      bt[j] = bi;
      nd[j] = bv + acf[t][lagMin + j];
    }
    back.push(bt);
    D = nd;
  }
  const path = new Int32Array(nw);
  let bi = 0;
  for (let j = 1; j < nl; j++) if (D[j] > D[bi]) bi = j;
  path[nw - 1] = bi;
  for (let t = nw - 1; t > 0; t--) path[t - 1] = back[t - 1][path[t]];
  const hr = new Float64Array(nw), q = new Float64Array(nw);
  for (let t = 0; t < nw; t++) {
    const i = path[t] + lagMin;
    let lag = i > 0 && i < acf[t].length - 1 ? parabolicPeak(acf[t], i) : i;
    lag = Math.min(Math.max(lag, i - 0.5), i + 0.5);
    hr[t] = (60 * fs) / lag;
    q[t] = acf[t][i];
  }
  return { hr, q };
}

export function bcgHeartRate(motion, fs) {
  const env = bcgEnvelopes(motion, fs);
  const { acf, centers, lagMin } = windowedAcf(env);
  const { hr, q } = trackHr(acf, lagMin);
  return { centers: Float64Array.from(centers), hr, q };
}

export const HEART_NAMES = ['hr', 'hr_std', 'hr_q'];

export function hrEpochFeatures(centers, hr, q, nEp, qMin = 0.25, ctxS = 60) {
  const F = Array.from({ length: nEp }, () => [NaN, NaN, NaN]);
  for (let e = 0; e < nEp; e++) {
    const c = (e + 0.5) * EPOCH_S;
    const qs = [], hs = [];
    for (let i = 0; i < centers.length; i++) {
      if (Math.abs(centers[i] - c) <= ctxS / 2) {
        qs.push(q[i]);
        if (Number.isFinite(hr[i]) && q[i] >= qMin) hs.push(hr[i]);
      }
    }
    F[e][2] = qs.length ? mean(qs) : 0;
    if (hs.length >= 3) { F[e][0] = median(hs); F[e][1] = std(hs); }
  }
  return F;
}

// ============================================================ mouvements
export const MOV_THR = 4.0;

export function activityPerSecond(chs, fs, lo = 1.0) {
  const hp = chs.map((c) => (lo ? filtfilt(c, fs, lo, null, 2) : c));
  const f = Math.round(fs);
  const n = Math.floor(hp[0].length / f);
  const out = new Float64Array(n);
  for (let s = 0; s < n; s++) {
    let e = 0;
    for (let i = s * f; i < (s + 1) * f; i++) for (const c of hp) e += c[i] * c[i];
    out[s] = Math.sqrt(e / f);
  }
  return out;
}

export function relativeActivity(...acts) {
  const n = Math.min(...acts.map((a) => a.length));
  const rel = new Float64Array(n);
  for (const a of acts) {
    const base = percentile(a.subarray(0, n), 10);
    const d = base > 0 ? base : 1;
    for (let i = 0; i < n; i++) rel[i] = Math.max(rel[i], a[i] / d);
  }
  return rel;
}

export const MOV_NAMES = ['mov_frac', 'mov_int', 't_since_mov', 't_to_mov'];

export function movEpochFeatures(rel, nEp, thr = MOV_THR) {
  const n = nEp * EPOCH_S;
  const r = new Float64Array(n).fill(1);
  r.set(rel.subarray(0, Math.min(n, rel.length)));
  const F = Array.from({ length: nEp }, () => [0, 0, 0, 0]);
  const any = new Array(nEp).fill(false);
  const moving = new Uint8Array(n);
  for (let e = 0; e < nEp; e++) {
    let cnt = 0, ex = 0;
    for (let i = e * EPOCH_S; i < (e + 1) * EPOCH_S; i++) {
      if (r[i] > thr) { cnt++; moving[i] = 1; }
      ex += Math.max(r[i] - thr, 0);
    }
    F[e][0] = cnt / EPOCH_S;
    F[e][1] = Math.log1p(ex);
    any[e] = cnt > 0;
  }
  const big = 120;
  let last = -1e9;
  for (let e = 0; e < nEp; e++) {
    if (any[e]) last = e;
    F[e][2] = Math.log1p(Math.min(big, ((e - last) * EPOCH_S) / 60));
  }
  let nxt = 1e9;
  for (let e = nEp - 1; e >= 0; e--) {
    if (any[e]) nxt = e;
    F[e][3] = Math.log1p(Math.min(big, ((nxt - e) * EPOCH_S) / 60));
  }
  return { F, moving };
}

// ============================================================ assemblage
export const BASE = [...RESP_NAMES, ...HEART_NAMES, ...MOV_NAMES];
const NORM = ['rr', 'rr_spec', 'ibi_cv', 'amp_cv', 'reg', 'spec_conc', 'hr', 'hr_std'];
const CTX = ['rr_z', 'ibi_cv', 'amp_cv', 'reg', 'spec_conc', 'amp_log', 'hr_z', 'hr_std', 'mov_frac'];
const CTX_WIN = [5, 11, 21];

export function rolling(x, w, fn) {
  const n = x.length, h = w >> 1;
  const c1 = new Float64Array(n + 1), c2 = new Float64Array(n + 1), cn = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) {
    const ok = Number.isFinite(x[i]);
    const v = ok ? x[i] : 0;
    c1[i + 1] = c1[i] + v; c2[i + 1] = c2[i] + v * v; cn[i + 1] = cn[i] + (ok ? 1 : 0);
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - h), hi = Math.min(n, i + h + 1);
    const cnt = cn[hi] - cn[lo];
    const m = cnt > 0 ? (c1[hi] - c1[lo]) / cnt : NaN;
    if (fn === 'mean') out[i] = m;
    else out[i] = cnt > 1 ? Math.sqrt(Math.max((c2[hi] - c2[lo]) / cnt - m * m, 0)) : NaN;
  }
  return out;
}

export function assemble(respF, hrF, movF) {
  const n = Math.min(respF.length, hrF.length, movF.length);
  const cols = {};
  BASE.forEach((name, j) => {
    const src = j < 8 ? respF : j < 11 ? hrF : movF;
    const jj = j < 8 ? j : j < 11 ? j - 8 : j - 11;
    cols[name] = Float64Array.from({ length: n }, (_, i) => src[i][jj]);
  });
  const out = { ...cols };
  for (const name of NORM) {
    const v = cols[name];
    const fin = Array.from(v).filter(Number.isFinite);
    const med = fin.length ? median(fin) : 0;
    const sc = robustScale(v);
    out[name + '_z'] = v.map((x) => (x - med) / sc);
  }
  const amp = cols.amp;
  const pos = Array.from(amp).filter((a) => a > 0);
  const medAmp = pos.length ? median(pos) : 1;
  out.amp_log = amp.map((a) => Math.log(Math.max(a, 1e-12) / medAmp));
  const tH = Float64Array.from({ length: n }, (_, i) => ((i + 0.5) * 30) / 3600);
  out.t_hours = tH;
  out.t_frac = tH.map((t) => t / Math.max(tH[n - 1], 1e-6));
  for (const name of CTX) {
    for (const w of CTX_WIN) {
      out[`${name}_m${w}`] = rolling(out[name], w, 'mean');
      out[`${name}_s${w}`] = rolling(out[name], w, 'std');
    }
  }
  return out; // dictionnaire nom → Float64Array (n époques)
}

// ============================================================ chaîne iPhone complète
/** acc, gyro : [x, y, z] Float64Array (m/s², rad/s) à fs Hz. */
export function phoneFeatures(acc, gyro, fs = 50) {
  const nEp = Math.floor(acc[0].length / fs / EPOCH_S);
  const rc = motionToRespChannels(acc, gyro, fs);
  const resp = analyzeResp(rc);
  const motion6 = [...acc, ...gyro.map((g) => g.map((v) => (v * G) / 10))];
  const heart = bcgHeartRate(motion6, fs);
  const hF = hrEpochFeatures(heart.centers, heart.hr, heart.q, nEp);
  const act = relativeActivity(activityPerSecond(acc, fs), activityPerSecond(gyro, fs));
  const { F: mF, moving } = movEpochFeatures(act, nEp);
  const feats = assemble(resp.F.slice(0, nEp), hF, mF);
  return { feats, nEp, resp, heart, activity: act, moving };
}

/** Estimation « en direct » (vérification du placement, écran de nuit) : ACP sur tout
 *  l'extrait (bords de filtrage exclus) puis pic spectral sur les dernières winS secondes. */
export function liveRespRate(acc, gyro, fs, winS = 60) {
  // sur un extrait court, la dérive basse fréquence (intégration du gyroscope) domine :
  // bande resserrée 0,13–0,7 Hz (8–42 resp/min) avant l'ACP
  const rc = motionToRespChannels(acc, gyro, fs).map((x) => filtfilt(x, FS_R, 0.13, 0.7, 2));
  const n = rc[0].length, c = rc.length;
  const a = Math.min(n, 5 * FS_R), b = Math.max(a, n - 3 * FS_R);
  if (b - a < 25 * FS_R) return { rr: null, conc: 0, wave: new Float64Array(0) };
  const cov = Array.from({ length: c }, () => new Array(c).fill(0));
  for (let i = 0; i < c; i++) for (let j = i; j < c; j++) {
    let s = 0;
    for (let k = a; k < b; k++) s += rc[i][k] * rc[j][k];
    cov[i][j] = cov[j][i] = s / (b - a);
  }
  const { vals, vecs } = eighSym(cov);
  let best = 0;
  for (let i = 1; i < c; i++) if (vals[i] > vals[best]) best = i;
  const w = new Float64Array(b - a);
  for (let k = a; k < b; k++) { let s = 0; for (let i = 0; i < c; i++) s += rc[i][k] * vecs[i][best]; w[k - a] = s; }
  const seg = w.slice(Math.max(0, w.length - winS * FS_R));
  // tendance linéaire retirée : sur un extrait court, la dérive basse fréquence fausserait le pic
  const m = seg.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < m; i++) { sx += i; sy += seg[i]; sxx += i * i; sxy += i * seg[i]; }
  const slope = (m * sxy - sx * sy) / (m * sxx - sx * sx), icpt = (sy - slope * sx) / m;
  for (let i = 0; i < m; i++) seg[i] -= icpt + slope * i;
  const nfft = 1024;
  const p = hannSpectrum(seg, nfft);
  let bk = -1, bv = -1, tot = 0;
  for (let k = 0; k < p.length; k++) {
    const f = (k * FS_R) / nfft;
    if (f >= 0.13 && f <= 0.7) { tot += p[k]; if (p[k] > bv) { bv = p[k]; bk = k; } }
  }
  if (tot <= 0) return { rr: null, conc: 0, wave: seg };
  let near = 0;
  for (let k = 0; k < p.length; k++) if (Math.abs(k - bk) * FS_R / nfft <= 0.04) near += p[k];
  return { rr: (60 * parabolicPeak(p, bk) * FS_R) / nfft, conc: near / tot, wave: seg };
}
