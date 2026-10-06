// Contrôleur de l'application Somno.
import { Recorder, requestMotionPermission, FS } from './recorder.js';
import * as store from './storage.js';
import { analyzeNight, loadModel, stageProba, STAGES } from './night.js';
import { phoneFeatures, liveRespRate, bcgHeartRate } from './analysis.js';
import { median } from './dsp.js';
import * as charts from './charts.js';
import { exportCsv, exportJson, exportRaw, playClip } from './export.js';
import { importSomno } from './import.js';

const $ = (s) => document.querySelector(s);
const G = 9.80665;
let recorder = null, checker = null, checkTimer = null, nightTimer = null, alarm = null;
let current = null; // rapport affiché

// ------------------------------------------------------------------ navigation
function go(id) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.toggle('active', s.id === id));
  document.body.classList.toggle('night-mode', id === 'night');
  window.scrollTo(0, 0);
  if (id === 'home') renderHome();
  if (id === 'history') renderHistory();
}
document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-go]');
  if (!b) return;
  if (b.dataset.go !== 'setup') stopCheck();
  go(b.dataset.go);
  if (b.dataset.check) $('#btn-check').scrollIntoView({ behavior: 'smooth', block: 'center' });
});

function toast(msg, ms = 3500) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add('hidden'), ms);
}

const fmtDur = (min) => {
  if (min == null || !Number.isFinite(min)) return '–';
  const h = Math.floor(min / 60), m = Math.round(min % 60);
  return h ? `${h} h ${String(m).padStart(2, '0')}` : `${m} min`;
};
const hhmm = (t) => new Date(t).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
const dayLabel = (t) => new Date(t).toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long' });

// ------------------------------------------------------------------ accueil
async function renderHome() {
  const nights = await store.listNights().catch(() => []);
  const box = $('#last-night');
  const interrupted = nights.find((n) => n.status === 'recording' && (!recorder || recorder.night?.id !== n.id));
  const last = nights.find((n) => n.report);
  if (interrupted) {
    box.innerHTML = `<div class="card-title">Nuit interrompue</div>
      <p>L'enregistrement du ${dayLabel(interrupted.start)} s'est arrêté sans être terminé (page fermée ?).</p>
      <button class="secondary" id="btn-recover">Analyser ce qui a été enregistré</button>`;
    box.classList.remove('hidden');
    $('#btn-recover').onclick = () => runAnalysis(interrupted.id);
  } else if (last) {
    const m = last.report.metrics;
    box.innerHTML = `<div class="card-title">Dernière nuit <small>${dayLabel(last.start)}</small></div>
      <canvas class="mini" style="width:100%;height:40px;display:block"></canvas>
      <div class="stats" style="margin-top:10px">
        <div class="stat"><b>${fmtDur(m.tstMin)}</b><small>de sommeil</small></div>
        <div class="stat"><b>${Math.round(m.efficiency * 100)} %</b><small>efficacité</small></div>
      </div>`;
    box.classList.remove('hidden');
    charts.miniHyp(box.querySelector('canvas'), last.report.hyp);
    box.onclick = () => showReport(last.report);
  } else box.classList.add('hidden');
}

// ------------------------------------------------------------------ préparation
$('#opt-alarm').addEventListener('change', (e) => $('#alarm-cfg').classList.toggle('hidden', !e.target.checked));
document.querySelectorAll('input[name=place]').forEach((r) => r.addEventListener('change', () => {
  const ns = $('input[name=place]:checked').value === 'nightstand';
  $('#opt-sonar').checked = ns;
  if (ns) $('#opt-audio').checked = true;
}));

function options() {
  return {
    place: $('input[name=place]:checked').value,
    audio: $('#opt-audio').checked || $('#opt-sonar').checked,
    clips: $('#opt-clips').checked,
    sonar: $('#opt-sonar').checked,
    alarm: $('#opt-alarm').checked ? { from: $('#alarm-from').value, to: $('#alarm-to').value } : null,
  };
}

$('#btn-check').addEventListener('click', async () => {
  if (checker) { stopCheck(); return; }
  if (!(await requestMotionPermission())) { toast('Accès aux capteurs de mouvement refusé.'); return; }
  checker = new Recorder({});
  await checker.start({ dry: true, audio: false });
  $('#btn-check').textContent = 'Arrêter la vérification';
  const t0 = Date.now();
  checkTimer = setInterval(() => liveCheck(t0), 2000);
});

function stopCheck() {
  clearInterval(checkTimer);
  checkTimer = null;
  if (checker) checker.stop();
  checker = null;
  $('#btn-check').textContent = 'Lancer la vérification';
}

/** Analyse rapide des dernières secondes : respiration, pouls, qualité capteurs. */
function liveSignal(rec, seconds) {
  const snap = rec.liveSnapshot(seconds);
  if (snap.n < FS * 35) return { ready: false, n: snap.n };
  const r = liveRespRate(snap.acc, snap.gyro, FS, 60);
  const motion6 = [...snap.acc, ...snap.gyro.map((g) => g.map((v) => (v * G) / 10))];
  const h = bcgHeartRate(motion6, FS);
  const recent = [];
  for (let i = Math.max(0, h.hr.length - 6); i < h.hr.length; i++) if (h.q[i] >= 0.3) recent.push(h.hr[i]);
  return {
    ready: true, n: snap.n, wave: r.wave.slice(-300), rr: r.rr, conc: r.conc,
    hr: recent.length >= 3 ? median(recent) : null, hrQ: median(Array.from(h.q).slice(-6)),
  };
}

function liveCheck(t0) {
  const s = checker.motionStats();
  $('#chk-rate').textContent = s.rateHz ? Math.round(s.rateHz) : '–';
  $('#check-time').textContent = `${Math.round((Date.now() - t0) / 1000)} s`;
  const L = liveSignal(checker, 120);
  const msg = [];
  if (s.rateHz && s.rateHz < 40) msg.push('⚠︎ Capteurs lents : fermez les autres applications.');
  if (s.stepAcc && s.stepAcc > 0.02) msg.push(`⚠︎ Accéléromètre arrondi par Safari (pas ${s.stepAcc.toFixed(3)} m/s²) : le pouls sera difficile à mesurer.`);
  if (!L.ready) {
    msg.push(`Collecte… (${Math.round(L.n / FS)} / 35 s)`);
  } else {
    charts.wave($('#check-wave'), L.wave);
    $('#chk-rr').textContent = L.rr ? Math.round(L.rr) : '–';
    $('#chk-hr').textContent = L.hr ? Math.round(L.hr) : '–';
    if (!L.rr || L.conc < 0.35) msg.push('Respiration peu nette : rapprochez l\'iPhone du thorax ou glissez-le sous le drap au niveau de la poitrine.');
    else msg.push('✓ Respiration détectée.');
    if (!L.hr) msg.push('Battements cardiaques non détectés : essayez sous l\'oreiller ou directement sous le thorax (facultatif — les stades restent estimés).');
    else msg.push('✓ Battements détectés.');
  }
  $('#check-msg').textContent = msg.join(' ');
}

// ------------------------------------------------------------------ nuit
$('#btn-start').addEventListener('click', async () => {
  stopCheck();
  const err = $('#setup-err');
  err.classList.add('hidden');
  try {
    if (!(await requestMotionPermission())) throw new Error('Accès aux capteurs de mouvement refusé (Réglages › Safari › Mouvement et orientation).');
    const opts = options();
    recorder = new Recorder({
      onStatus: (s) => { if (s.audioError) toast('Micro indisponible : ' + s.audioError); },
      onLive: (l) => { if (l.sonar) recorder.sonarDist = l.sonar.distM; },
    });
    await recorder.start(opts);
    if (opts.alarm) setupAlarm(opts.alarm);
    go('night');
    nightTimer = setInterval(tickNight, 1000);
    tickNight();
  } catch (e) {
    err.textContent = e.message || String(e);
    err.classList.remove('hidden');
  }
});

let lastStatus = 0;
function tickNight() {
  const now = new Date();
  $('#clock').textContent = hhmm(now);
  const el = (Date.now() - recorder.night.start) / 1000;
  $('#elapsed').textContent = `${String(Math.floor(el / 3600)).padStart(2, '0')}:${String(Math.floor((el % 3600) / 60)).padStart(2, '0')}`;
  if (Date.now() - lastStatus > 30000 && recorder.nTotal > FS * 40) {
    lastStatus = Date.now();
    const L = liveSignal(recorder, 90);
    const parts = [];
    if (L.ready && L.rr) parts.push(`${Math.round(L.rr)} resp/min`);
    if (L.ready && L.hr) parts.push(`${Math.round(L.hr)} bpm`);
    if (recorder.sonarDist) parts.push(`sonar ${recorder.sonarDist.toFixed(2)} m`);
    if (recorder.night.gaps.length) parts.push(`${recorder.night.gaps.length} interruption(s)`);
    $('#night-status').textContent = parts.join(' · ') || 'Enregistrement…';
  }
  if (alarm) checkAlarm(now);
}

// affichage des commandes au toucher, arrêt par appui long (évite un arrêt involontaire)
$('#night').addEventListener('touchstart', () => {
  $('#night-controls').classList.add('show');
  clearTimeout(tickNight.hide);
  tickNight.hide = setTimeout(() => $('#night-controls').classList.remove('show'), 5000);
}, { passive: true });

const stopBtn = $('#btn-stop');
let holdStart = null, holdRaf = null;
const holdTick = () => {
  const p = Math.min(1, (Date.now() - holdStart) / 1500);
  stopBtn.style.setProperty('--p', `${p * 100}%`);
  if (p >= 1) { holdEnd(); endNight(); return; }
  holdRaf = requestAnimationFrame(holdTick);
};
const holdEnd = () => { cancelAnimationFrame(holdRaf); holdStart = null; stopBtn.style.setProperty('--p', '0%'); };
stopBtn.addEventListener('pointerdown', () => { holdStart = Date.now(); holdTick(); });
['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) => stopBtn.addEventListener(ev, () => { if (holdStart) holdEnd(); }));

async function endNight() {
  clearInterval(nightTimer);
  stopAlarmSound();
  alarm = null;
  const id = await recorder.stop();
  recorder = null;
  await runAnalysis(id);
}

async function runAnalysis(id) {
  go('analyzing');
  try {
    const rep = await analyzeNight(id, (m) => { $('#progress').textContent = m; });
    notifyNative(rep);
    showReport(rep);
  } catch (e) {
    console.error(e);
    toast('Analyse impossible : ' + (e.message || e), 6000);
    go('home');
  }
}

// ------------------------------------------------------------------ réveil intelligent
function setupAlarm(cfg) {
  const toDate = (hm) => {
    const [h, m] = hm.split(':').map(Number);
    const d = new Date();
    d.setHours(h, m, 0, 0);
    if (d < new Date()) d.setDate(d.getDate() + 1);
    return d;
  };
  const to = toDate(cfg.to);
  let from = toDate(cfg.from);
  if (from > to) from = new Date(from.getTime() - 86400e3);
  alarm = { from, to, rung: false, lastEval: 0 };
  // contexte audio créé pendant le geste de l'utilisateur (exigence iOS)
  alarm.ctx = new (window.AudioContext || window.webkitAudioContext)();
  $('#alarm-info').textContent = `Réveil entre ${hhmm(from)} et ${hhmm(to)}. En mode silencieux l'alarme web peut rester muette : gardez une alarme de secours.`;
}

async function checkAlarm(now) {
  if (alarm.rung || now < alarm.from) return;
  if (now >= alarm.to) return ringAlarm('Fin de la fenêtre de réveil');
  if (Date.now() - alarm.lastEval < 30000) return;
  alarm.lastEval = Date.now();
  const snap = recorder.liveSnapshot(600);
  if (snap.n < FS * 300) return;
  const r = phoneFeatures(snap.acc, snap.gyro, FS);
  const model = await loadModel();
  const t = r.nEp - 1;
  const p = stageProba(model, r.feats, r.nEp)[t];
  const moved = r.feats.mov_frac[t] > 0 || r.feats.mov_frac[t - 1] > 0;
  if (p[0] + p[1] > 0.6 && moved) ringAlarm('Sommeil léger détecté');
}

function ringAlarm(reason) {
  alarm.rung = true;
  $('#night-status').textContent = `⏰ ${reason} — bonjour !`;
  $('#night-controls').classList.add('show');
  const ctx = alarm.ctx;
  ctx.resume();
  const g = ctx.createGain();
  g.gain.value = 0;
  g.connect(ctx.destination);
  const notes = [523.25, 659.25, 783.99];
  let k = 0;
  alarm.timer = setInterval(() => {
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = notes[k++ % 3];
    o.connect(g);
    const t = ctx.currentTime;
    const vol = Math.min(0.8, 0.1 + k * 0.02);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.05);
    g.gain.linearRampToValueAtTime(0, t + 0.6);
    o.start(t); o.stop(t + 0.65);
  }, 900);
  if (navigator.vibrate) navigator.vibrate([400, 200, 400]);
}

function stopAlarmSound() {
  if (alarm?.timer) clearInterval(alarm.timer);
  alarm?.ctx?.close?.();
}

// ------------------------------------------------------------------ rapport
function stat(value, label) { return `<div class="stat"><b>${value}</b><small>${label}</small></div>`; }

async function showReport(rep) {
  current = rep;
  go('report');
  $('#rep-title').textContent = new Date(rep.start).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short' });
  const m = rep.metrics;
  const colors = charts.STAGE_COLORS();
  const stageBar = [3, 1, 2].map((s) => `<i style="flex:${m.stageMin[s]};background:${colors[s]}"></i>`).join('');
  const q = rep.quality;
  const warn = [];
  if (q.respCoverage < 0.6) warn.push('Respiration mal captée sur une partie de la nuit : stades moins fiables.');
  if (q.hrCoverage < 0.3) warn.push('Pouls rarement détecté (iPhone loin du thorax ?) : stades estimés surtout par la respiration et les mouvements.');
  if (q.gaps) warn.push(`${q.gaps} interruption(s) d'enregistrement.`);
  const clipsHtml = rep.clips.length ? rep.clips.slice(0, 40).map((c) =>
    `<div class="clip"><span>${hhmm(rep.start + c.t * 1000)}</span><button data-clip="${c.t}">▶︎ Écouter</button></div>`).join('') : '<p class="note">Aucun bruit marquant.</p>';
  $('#rep-body').innerHTML = `
    <p class="span">${dayLabel(rep.start)} · couché ${hhmm(rep.start)} → levé ${hhmm(rep.end || rep.start + rep.nEp * 30e3)}</p>
    <div class="card">
      <div class="card-title">Hypnogramme <small>${rep.source}</small></div>
      <canvas id="c-hyp" class="hyp"></canvas>
      <div class="legend"><span><i style="background:${colors[0]}"></i>Éveil</span><span><i style="background:${colors[3]}"></i>REM</span><span><i style="background:${colors[1]}"></i>Léger</span><span><i style="background:${colors[2]}"></i>Profond</span><span>· teinte pâle = incertain</span></div>
    </div>
    <div class="card">
      <div class="stats">
        ${stat(fmtDur(m.tstMin), 'temps de sommeil')}
        ${stat(Math.round(m.efficiency * 100) + ' %', 'efficacité')}
        ${stat(fmtDur(m.sleepOnsetMin), 'pour s\'endormir')}
        ${stat(`${fmtDur(m.wasoMin)}`, `éveillé pendant la nuit · ${m.awakenings} réveil(s)`)}
      </div>
      <div class="stages-bar">${stageBar}</div>
      <div class="stats">
        ${stat(fmtDur(m.stageMin[2]), `profond · ${Math.round(m.stagePct[2] * 100)} %`)}
        ${stat(fmtDur(m.stageMin[3]), `REM · ${Math.round(m.stagePct[3] * 100)} %`)}
        ${stat(fmtDur(m.stageMin[1]), `léger · ${Math.round(m.stagePct[1] * 100)} %`)}
        ${stat(m.movementsPerHour.toFixed(1), 'mouvements / heure')}
      </div>
    </div>
    <div class="card">
      <div class="card-title">Respiration <small>${m.rrMean ? m.rrMean.toFixed(1) + ' /min en moyenne' : ''}</small></div>
      <canvas id="c-rr" class="series"></canvas>
      <p class="note">Événements respiratoires (chute d'amplitude ≥ 50 % pendant ≥ 10 s) : <b>${m.rdi.toFixed(1)} / h</b> (${m.respEvents}).
      ${m.rdi >= 15 ? '<span class="warn">Valeur élevée : si vous ronflez, êtes fatigué le jour ou avez des pauses respiratoires, parlez-en à un médecin.</span>' : ''}
      Indicateur de dépistage, pas un diagnostic.</p>
    </div>
    <div class="card">
      <div class="card-title">Cœur <small>${m.hrMean ? Math.round(m.hrMean) + ' bpm en moyenne · min ' + Math.round(m.hrMin) : 'non mesuré'}</small></div>
      <canvas id="c-hr" class="series"></canvas>
    </div>
    ${rep.series.db ? `<div class="card">
      <div class="card-title">Sons <small>${m.snoreMin ? fmtDur(m.snoreMin) + ' de ronflement' : 'pas de ronflement détecté'}</small></div>
      <canvas id="c-db" class="series"></canvas>
      <div style="margin-top:8px">${clipsHtml}</div>
    </div>` : ''}
    <div class="card">
      <div class="card-title">Mouvements</div>
      <canvas id="c-mov" class="series"></canvas>
    </div>
    <div class="card">
      <div class="card-title">Qualité du signal</div>
      <p class="note">Respiration exploitable ${Math.round(q.respCoverage * 100)} % du temps · pouls ${Math.round(q.hrCoverage * 100)} % · capteurs ${q.motionRateHz ? Math.round(q.motionRateHz) + ' Hz' : '–'}${rep.sonar ? ` · sonar à ${rep.sonar.distM?.toFixed(2)} m` : ''}</p>
      ${warn.map((w) => `<p class="note warn">${w}</p>`).join('')}
    </div>
    <button class="secondary" id="btn-del">Supprimer cette nuit</button>`;
  const start = rep.start;
  charts.hypnogram($('#c-hyp'), rep.hyp, start, { proba: rep.proba });
  charts.series($('#c-rr'), rep.series.rr, start, { color: '#5fa8ff', unit: '' });
  charts.series($('#c-hr'), rep.series.hr, start, { color: '#f26d6d', unit: '' });
  if (rep.series.db) charts.series($('#c-db'), rep.series.db, start, { color: '#b48cff', unit: ' dB' });
  charts.series($('#c-mov'), rep.series.mov, start, { color: '#f2b84b', bars: true, min: 0, max: 1 });
  document.querySelectorAll('[data-clip]').forEach((b) => b.addEventListener('click', () => playClip(rep.id, Number(b.dataset.clip))));
  $('#btn-del').onclick = async () => {
    if (!confirm('Supprimer définitivement cette nuit ?')) return;
    await store.deleteNight(rep.id);
    go('history');
  };
}

$('#btn-export').addEventListener('click', () => {
  if (!current) return;
  const box = document.createElement('div');
  box.className = 'card export-list';
  box.innerHTML = `<div class="card-title">Exporter</div>
    <button data-x="json">Rapport (JSON)</button>
    <button data-x="csv">Époques de 30 s (CSV)</button>
    <button data-x="raw">Données brutes (.somno, pour l'analyse Python)</button>`;
  $('#rep-body').prepend(box);
  box.addEventListener('click', async (e) => {
    const x = e.target.dataset.x;
    if (x === 'json') await exportJson(current);
    if (x === 'csv') await exportCsv(current);
    if (x === 'raw') { toast('Préparation du fichier…'); await exportRaw(current.id); }
    box.remove();
  });
});

// ------------------------------------------------------------------ historique
async function renderHistory() {
  const nights = await store.listNights();
  const done = nights.filter((n) => n.report);
  charts.trend($('#trend'), done.slice(0, 14).reverse().map((n) => ({ start: n.start, metrics: n.report.metrics })));
  $('#hist-list').innerHTML = nights.map((n) => `
    <button class="card hist-item" data-id="${n.id}">
      <div class="hist-head"><b>${dayLabel(n.start)}</b><small>${n.report ? fmtDur(n.report.metrics.tstMin) : n.status === 'recording' ? 'interrompue' : 'à analyser'}</small></div>
      ${n.report ? '<canvas class="mini"></canvas>' : ''}
    </button>`).join('') || '<p class="fine">Aucune nuit enregistrée.</p>';
  document.querySelectorAll('.hist-item').forEach((el) => {
    const n = nights.find((x) => x.id === el.dataset.id);
    if (n.report) charts.miniHyp(el.querySelector('canvas'), n.report.hyp);
    el.onclick = () => (n.report ? showReport(n.report) : runAnalysis(n.id));
  });
}

// ------------------------------------------------------------------ import & app iOS native
async function importAndAnalyze(buffer) {
  go('analyzing');
  $('#progress').textContent = 'Import…';
  const id = await importSomno(buffer);
  await runAnalysis(id);
}

$('#file-import')?.addEventListener('change', async (e) => {
  const f = e.target.files?.[0];
  if (f) await importAndAnalyze(await f.arrayBuffer()).catch((er) => toast('Import impossible : ' + er.message));
});

// l'app iOS ouvre index.html?import=<url> puis reçoit le rapport (→ Apple Santé)
const nativeBridge = window.webkit?.messageHandlers?.somno;
const importUrl = new URLSearchParams(location.search).get('import');
if (importUrl) {
  fetch(importUrl).then((r) => r.arrayBuffer()).then(importAndAnalyze)
    .catch((e) => toast('Import impossible : ' + e.message, 6000));
}
export function notifyNative(rep) {
  try { nativeBridge?.postMessage({ type: 'report', report: { id: rep.id, start: rep.start, hyp: rep.hyp, metrics: rep.metrics } }); } catch { /* hors app native */ }
}

// ------------------------------------------------------------------ démarrage
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
window.addEventListener('beforeunload', (e) => { if (recorder) { e.preventDefault(); e.returnValue = ''; } });
renderHome();
loadModel().catch(() => toast('Modèle de stades introuvable.'));
export { STAGES };
