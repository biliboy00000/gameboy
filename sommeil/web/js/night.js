// Analyse du matin : données brutes de la nuit → hypnogramme, métriques, courbes.
import * as store from './storage.js';
import {
  phoneFeatures, assemble, analyzeResp, respChannelsFromSignal, hrEpochFeatures, windowedAcf,
  trackHr, relativeActivity, movEpochFeatures, EPOCH_S, FS_ENV,
} from './analysis.js';
import { filtfilt, resampleLinear, median, percentile, mean } from './dsp.js';
import { predictProba, argmax } from './model.js';
import { predictSeq } from './seqmodel.js';
import { detectEvents, rdi } from './apnea.js';
import { sonarConsts } from './sonar.js';

export const STAGES = ['Éveil', 'Léger', 'Profond', 'REM'];
let modelCache = null;

export async function loadModel() {
  if (!modelCache) modelCache = await (await fetch(new URL('../model/staging.json', import.meta.url))).json();
  return modelCache;
}

/** Probabilités des 4 stades : moyenne arbres boostés + GRU bidirectionnel. */
export function stageProba(model, feats, nEp) {
  const X = Array.from({ length: nEp }, (_, i) => model.hgb.features.map((n) => feats[n]?.[i]));
  const Ph = predictProba(model.hgb, X);
  if (!model.gru) return Ph;
  const Pg = predictSeq(model.gru, feats, nEp);
  const w = model.w_gru ?? 0.5;
  return Ph.map((p, i) => p.map((v, k) => (1 - w) * v + w * Pg[i][k]));
}

async function loadMotion(id) {
  const chunks = (await store.getChunks(id, 'motion')).sort((a, b) => a.seq - b.seq);
  const n = chunks.reduce((s, c) => s + c.n, 0);
  const acc = [0, 1, 2].map(() => new Float64Array(n)), gyro = [0, 1, 2].map(() => new Float64Array(n));
  let k = 0;
  for (const c of chunks) {
    for (let i = 0; i < c.n; i++, k++) {
      for (let j = 0; j < 3; j++) { acc[j][k] = c.data[i * 6 + j]; gyro[j][k] = c.data[i * 6 + 3 + j]; }
    }
  }
  return { acc, gyro, n };
}

async function loadSonar(id, fs) {
  const chunks = (await store.getChunks(id, 'sonar')).sort((a, b) => a.seq - b.seq);
  if (!chunks.length) return null;
  const { ne, frameRate } = sonarConsts(fs);
  const nEp = chunks[chunks.length - 1].seq + 1;
  const disp = new Float64Array(nEp * ne), act = new Float64Array(nEp * ne);
  const dist = new Float64Array(nEp).fill(NaN);
  for (const c of chunks) {
    disp.set(c.data.subarray(0, c.n), c.seq * ne);
    act.set(c.data.subarray(c.n, 2 * c.n), c.seq * ne);
    dist[c.seq] = c.distM;
  }
  return { disp, act, dist, frameRate, nEp, kind: 'sonar' };
}

/** Profondeur TrueDepth (app iOS) : grille de cellules (mm) → déplacement de la
 *  cellule la plus « respiratoire » à chaque époque (même critère que le sonar). */
async function loadDepth(id) {
  const ch = (await store.getChunks(id, 'depth'))[0];
  if (!ch) return null;
  const rate = ch.rate, cols = ch.data.length / ch.n;
  const ne = Math.round(EPOCH_S * rate), nEp = Math.floor(ch.n / ne);
  const disp = new Float64Array(nEp * ne), act = new Float64Array(nEp * ne);
  let nfft = 1; while (nfft < 2 * ne) nfft <<= 1;
  const { fft } = await import('./dsp.js');
  for (let e = 0; e < nEp; e++) {
    const a = Math.max(0, e * ne - ne / 2), b = Math.min(ch.n, (e + 1) * ne + ne / 2);
    let best = 0, bestS = -1;
    for (let c = 0; c < cols; c++) {
      const re = new Float64Array(nfft), im = new Float64Array(nfft);
      let m = 0;
      for (let t = a; t < b; t++) m += ch.data[t * cols + c];
      m /= b - a;
      for (let t = a; t < b; t++) re[t - a] = ch.data[t * cols + c] - m;
      fft(re, im);
      let pb = 0, ph = 0;
      for (let k = 1; k < nfft / 2; k++) {
        const f = (k * rate) / nfft, p = re[k] * re[k] + im[k] * im[k];
        if (f >= 0.1 && f <= 0.7) pb += p; else if (f > 0.7 && f <= 5) ph += p;
      }
      const sc = (pb * pb) / (pb + ph + 1e-30);
      if (sc > bestS) { bestS = sc; best = c; }
    }
    let m = 0;
    for (let t = 0; t < ne; t++) m += ch.data[(e * ne + t) * cols + best];
    m /= ne;
    for (let t = 0; t < ne; t++) {
      const i = e * ne + t;
      disp[i] = ch.data[i * cols + best] - m;
      if (i > 0) { let s2 = 0; for (let c = 0; c < cols; c++) s2 += (ch.data[i * cols + c] - ch.data[(i - 1) * cols + c]) ** 2; act[i] = Math.sqrt(s2); }
    }
  }
  return { disp, act, frameRate: rate, nEp, dist: new Float64Array(nEp).fill(NaN), kind: 'profondeur TrueDepth' };
}

async function loadAudio(id) {
  const chunks = (await store.getChunks(id, 'audio')).sort((a, b) => a.seq - b.seq);
  if (!chunks.length) return null;
  const n = chunks.reduce((s, c) => s + c.n, 0);
  const v = new Float32Array(n * 8);
  let k = 0;
  for (const c of chunks) { v.set(c.data, k); k += c.data.length; }
  return { v, n, rate: 4 };
}

/** Analyse des sons (4 trames/s × [niveau, 6 bandes, crête]) par époque de 30 s. */
export function audioEpochs(audio, nEp, breathRate) {
  const out = [];
  if (!audio) return null;
  const { v, n } = audio;
  const lvl = new Float64Array(n), low = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    lvl[i] = v[i * 8];
    // énergie 150–1000 Hz (bandes 2 et 3), domaine des ronflements
    low[i] = 10 * Math.log10(10 ** (v[i * 8 + 2] / 10) + 10 ** (v[i * 8 + 3] / 10));
  }
  const bg = percentile(lvl, 20);
  const bgLow = percentile(low, 20);
  const per = 4 * EPOCH_S;
  for (let e = 0; e < nEp; e++) {
    const a = e * per, b = Math.min(n, (e + 1) * per);
    if (a >= n) { out.push(null); continue; }
    const seg = lvl.subarray(a, b), sl = low.subarray(a, b);
    // ronflement : bouffées graves répétées au rythme respiratoire (2–8 s)
    const x = Array.from(sl, (d) => Math.max(0, d - bgLow - 6));
    const m = mean(x);
    let best = 0;
    const per0 = breathRate?.[e] ? Math.round((60 / breathRate[e]) * 4) : null;
    const lags = per0 ? [per0 - 1, per0, per0 + 1] : Array.from({ length: 25 }, (_, i) => 8 + i);
    let den = 0;
    for (const y of x) den += (y - m) * (y - m);
    for (const L of lags) {
      if (L <= 0 || L >= x.length) continue;
      let s = 0;
      for (let i = 0; i + L < x.length; i++) s += (x[i] - m) * (x[i + L] - m);
      if (den > 0) best = Math.max(best, s / den);
    }
    const loudFrac = x.filter((y) => y > 0).length / x.length;
    const snore = best > 0.3 && loudFrac > 0.15 && percentile(sl, 90) - bgLow > 10;
    const events = Array.from(seg).filter((d) => d > bg + 15).length;
    out.push({ db: mean(seg), dbMax: Math.max(...seg), snore, snoreScore: best * loudFrac, events });
  }
  return { epochs: out, background: bg };
}

function sonarFeatures(sonar, nEp) {
  const rc = respChannelsFromSignal(sonar.disp, sonar.frameRate);
  const resp = analyzeResp(rc);
  // FC depuis le déplacement thoracique 0,8–3 Hz
  let x = filtfilt(sonar.disp, sonar.frameRate, 0.8, 3.0, 2);
  x = resampleLinear(x, sonar.frameRate, FS_ENV);
  const { acf, centers, lagMin } = windowedAcf([x]);
  const { hr, q } = trackHr(acf, lagMin);
  const hF = hrEpochFeatures(Float64Array.from(centers), hr, q, nEp);
  // mouvement : variation du profil de distance par seconde
  const fr = Math.round(sonar.frameRate);
  const ns = Math.floor(sonar.act.length / sonar.frameRate);
  const actS = new Float64Array(ns);
  for (let s = 0; s < ns; s++) {
    let acc = 0, c = 0;
    for (let i = Math.floor(s * sonar.frameRate); i < Math.floor((s + 1) * sonar.frameRate); i++) { acc += sonar.act[i] ** 2; c++; }
    actS[s] = Math.sqrt(acc / Math.max(1, c));
  }
  const rel = relativeActivity(actS);
  const { F: mF, moving } = movEpochFeatures(rel, nEp);
  return { resp, hF, mF, moving, hr: { centers, hr, q }, fr };
}

export async function analyzeNight(id, onProgress = () => {}) {
  const night = await store.getNight(id);
  onProgress('Chargement des données…');
  const motion = await loadMotion(id);
  const fs = night.fs || 50;
  const nEp = Math.floor(motion.n / fs / EPOCH_S);
  if (nEp < 20) throw new Error('Enregistrement trop court (moins de 10 minutes).');
  onProgress('Respiration, cœur et mouvements…');
  await new Promise((r) => setTimeout(r, 30));
  const ph = phoneFeatures(motion.acc, motion.gyro, fs);
  let feats = ph.feats;
  let source = 'accéléromètre + gyroscope';
  let respWave = ph.resp.w, breaths = ph.resp.times, heart = ph.heart, moving = ph.moving;

  const sonar = (night.opts?.sonar ? await loadSonar(id, night.audioRate || 48000) : null) || await loadDepth(id);
  let sonarInfo = null;
  if (sonar && sonar.nEp >= 20) {
    onProgress('Sonar…');
    await new Promise((r) => setTimeout(r, 30));
    const so = sonarFeatures(sonar, nEp);
    const conc = (F) => median(F.map((r) => r[5]));
    const cMotion = conc(ph.resp.F), cSonar = conc(so.resp.F);
    sonarInfo = { specConcMotion: cMotion, specConcSonar: cSonar, distM: median(Array.from(sonar.dist)) };
    // téléphone sur la table de nuit (accéléromètre muet) ou sonar plus net → sonar
    if (cSonar > cMotion) {
      const pad = (F, k) => Array.from({ length: nEp }, (_, i) => F[i] || new Array(k).fill(NaN));
      const useMotionMov = percentile(ph.activity, 90) > 2; // le téléphone sent-il les mouvements ?
      feats = assemble(pad(so.resp.F, 8), pad(so.hF, 3), useMotionMov ? movEpochFeaturesFrom(ph) : pad(so.mF, 4));
      source = sonar.kind + (useMotionMov ? ' + mouvements accéléromètre' : '');
      respWave = so.resp.w; breaths = so.resp.times;
      heart = { centers: Float64Array.from(so.hr.centers), hr: so.hr.hr, q: so.hr.q };
      if (!useMotionMov) moving = so.moving;
    }
  }

  onProgress('Stades de sommeil…');
  await new Promise((r) => setTimeout(r, 30));
  const model = await loadModel();
  const Ps = stageProba(model, feats, nEp);
  let hyp = Ps.map(argmax);
  // période de sommeil : de l'endormissement (5 min de sommeil sur 6) au dernier sommeil
  let onset = hyp.findIndex((_, i) => i + 12 <= nEp && hyp.slice(i, i + 12).filter((s) => s > 0).length >= 10);
  if (onset < 0) onset = nEp;
  let last = nEp - 1;
  while (last > onset && hyp[last] === 0) last--;
  hyp = hyp.map((s, i) => (i < onset || i > last ? 0 : s));

  onProgress('Sons et respiration…');
  const audio = night.opts?.audio ? await loadAudio(id) : null;
  const rrEp = feats.rr;
  const aud = audioEpochs(audio, nEp, rrEp);
  const movingS = moving;
  const events = detectEvents(respWave, movingS);
  const sleepMask = hyp.map((s) => s > 0);
  const resp = rdi(events, sleepMask);

  const m = metrics(hyp, feats, aud, night, onset, last);
  m.rdi = resp.rdi; m.respEvents = resp.n;
  const clips = await store.getClips(id);
  const report = {
    version: 1, id, start: night.start, end: night.end, nEp, hyp, proba: Ps.map((p) => p.map((v) => +v.toFixed(3))),
    source, sonar: sonarInfo, metrics: m,
    series: {
      rr: Array.from(feats.rr, (v) => (Number.isFinite(v) ? +v.toFixed(2) : null)),
      hr: Array.from(feats.hr, (v) => (Number.isFinite(v) ? +v.toFixed(1) : null)),
      mov: Array.from(feats.mov_frac, (v) => +v.toFixed(3)),
      snore: aud ? aud.epochs.map((a) => (a?.snore ? 1 : 0)) : null,
      db: aud ? aud.epochs.map((a) => (a ? +a.db.toFixed(1) : null)) : null,
    },
    events: events.map(([s, e, d]) => [Math.round(s), Math.round(e), +d.toFixed(2)]),
    clips: clips.map((c) => ({ t: c.t, rate: c.rate, n: c.data.length })),
    quality: quality(feats, night, ph),
    breathCount: breaths.length,
  };
  night.report = report;
  night.status = 'analyzed';
  await store.putNight(night);
  return report;
}

function movEpochFeaturesFrom(ph) {
  const nEp = ph.nEp;
  return Array.from({ length: nEp }, (_, i) => ['mov_frac', 'mov_int', 't_since_mov', 't_to_mov'].map((k) => ph.feats[k][i]));
}

function metrics(hyp, feats, aud, night, onset, last) {
  const n = hyp.length;
  const min = (k) => k / 2; // époques → minutes
  const sleep = hyp.filter((s) => s > 0).length;
  const inPeriod = Math.max(0, last - onset + 1);
  const wasoEp = hyp.slice(onset, last + 1).filter((s) => s === 0).length;
  let awakenings = 0;
  for (let i = onset + 1; i <= last; i++) if (hyp[i] === 0 && hyp[i - 1] > 0) {
    let j = i; while (j <= last && hyp[j] === 0) j++;
    if (j - i >= 2) awakenings++; // ≥ 1 min
  }
  const stageMin = [0, 1, 2, 3].map((s) => min(hyp.filter((h) => h === s).length));
  const sel = (arr) => Array.from(arr).filter((v, i) => hyp[i] > 0 && Number.isFinite(v));
  const rr = sel(feats.rr), hr = sel(feats.hr);
  let snoreMin = 0, soundEvents = 0;
  if (aud) aud.epochs.forEach((a, i) => { if (a && hyp[i] > 0 && a.snore) snoreMin += 0.5; if (a) soundEvents += a.events > 0 ? 1 : 0; });
  return {
    inBedMin: min(n), sleepOnsetMin: min(onset), tstMin: min(sleep),
    efficiency: n ? sleep / n : 0, wasoMin: min(wasoEp), awakenings,
    finalWakeMin: min(n - 1 - last),
    stageMin, stagePct: stageMin.map((v) => (sleep ? v / min(sleep) : 0)),
    sleepPeriodMin: min(inPeriod),
    rrMean: rr.length ? mean(rr) : null, rrMin: rr.length ? percentile(rr, 5) : null,
    hrMean: hr.length ? mean(hr) : null, hrMin: hr.length ? percentile(hr, 5) : null,
    movementsPerHour: (() => {
      let c = 0;
      for (let i = 1; i < n; i++) if (feats.mov_frac[i] > 0 && feats.mov_frac[i - 1] === 0) c++;
      return n ? c / (n / 120) : 0;
    })(),
    snoreMin, soundEpochs: soundEvents,
  };
}

function quality(feats, night, ph) {
  const n = feats.rr.length;
  const fin = (a) => Array.from(a).filter(Number.isFinite).length / Math.max(1, n);
  return {
    respCoverage: fin(feats.rr), hrCoverage: fin(feats.hr),
    specConc: median(Array.from(feats.spec_conc)), hrQuality: median(Array.from(feats.hr_q)),
    motionRateHz: night.motionStats?.rateHz, accStep: night.motionStats?.stepAcc, gyroStep: night.motionStats?.stepGyro,
    gaps: night.gaps?.length || 0,
  };
}
