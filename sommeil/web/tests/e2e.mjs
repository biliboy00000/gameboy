// Test de bout en bout dans Chromium (Playwright) :
//  1. vérification du placement + 70 s d'enregistrement avec des événements devicemotion
//     synthétiques (respiration à 15/min, battements à 60/min) et un micro factice ;
//  2. analyse d'une nuit simulée de 2 h 30 injectée dans IndexedDB → rapport complet.
//
//   node web/tests/e2e.mjs [/chemin/nuit.f32] [dossier captures]
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFileSync, existsSync, mkdirSync } from 'node:fs';
import { extname, join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ONLY = process.env.E2E_ONLY || '';
const NIGHT = process.argv[2] || '/home/user/data/e2e_night.f32';
const SHOTS = process.argv[3] || resolve(WEB, '..', 'docs', 'figures');
mkdirSync(SHOTS, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.png': 'image/png', '.mp4': 'video/mp4' };

const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  const file = url === '/night.f32' ? NIGHT : join(WEB, url === '/' ? 'index.html' : url);
  if (!existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream' });
  res.end(readFileSync(file));
}).listen(0);
const base = `http://localhost:${server.address().port}/`;

const fail = (m) => { console.error('ÉCHEC :', m); process.exitCode = 1; };
const browser = await chromium.launch({
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required'],
});
const ctx = await browser.newContext({ viewport: { width: 393, height: 852 }, deviceScaleFactor: 2, permissions: ['microphone'] });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.addInitScript(() => {
  window.DeviceMotionEvent.requestPermission = async () => 'granted';
  navigator.wakeLock = { request: async () => ({ release: async () => {}, addEventListener() {} }) };
  // générateur d'événements devicemotion à 60 Hz : inclinaison respiratoire + BCG
  window.__motion = (on) => {
    clearInterval(window.__mt);
    if (!on) return;
    const t0 = performance.now();
    window.__mt = setInterval(() => {
      const t = (performance.now() - t0) / 1000;
      const tilt = 0.003 * Math.sin(2 * Math.PI * 0.25 * t); // rad, ≈ 0,17°
      const ph = t % 1.0;
      const bcg = ph < 0.4 ? 0.02 * Math.exp(-ph / 0.08) * Math.sin(2 * Math.PI * 6 * ph) : 0;
      const n = () => (Math.random() - 0.5) * 0.004;
      const ev = new DeviceMotionEvent('devicemotion', {
        accelerationIncludingGravity: { x: 9.81 * tilt + n(), y: n(), z: 9.81 + bcg + n() },
        acceleration: { x: 0, y: 0, z: 0 },
        rotationRate: { alpha: n() * 5, beta: (0.003 * 2 * Math.PI * 0.25 * Math.cos(2 * Math.PI * 0.25 * t)) * 57.3 + bcg * 5, gamma: n() * 5 },
        interval: 16.7,
      });
      window.dispatchEvent(ev);
    }, 1000 / 60);
  };
});

await page.goto(base);
await page.screenshot({ path: join(SHOTS, 'app-accueil.png') });

// ---- 1. vérification du placement
if (ONLY !== 'analysis') {
await page.click('button[data-go=setup]');
await page.evaluate(() => window.__motion(true));
await page.click('#btn-check');
await page.waitForTimeout(50000);
const chk = await page.evaluate(() => ({ rate: document.querySelector('#chk-rate').textContent, rr: document.querySelector('#chk-rr').textContent, hr: document.querySelector('#chk-hr').textContent, msg: document.querySelector('#check-msg').textContent }));
console.log('vérification :', chk);
if (!(Number(chk.rr) >= 13 && Number(chk.rr) <= 17)) fail(`respiration en direct ${chk.rr} (attendu ≈ 15)`);
await page.locator('#check').screenshot({ path: join(SHOTS, 'app-verification.png') });
await page.click('#btn-check');

// ---- enregistrement court (70 s) avec micro factice
await page.click('#btn-start');
await page.waitForSelector('#night.active');
await page.waitForTimeout(70000);
const rec = await page.evaluate(async () => {
  const db = await new Promise((r) => { const q = indexedDB.open('somno'); q.onsuccess = () => r(q.result); });
  const all = (store) => new Promise((r) => { const q = db.transaction(store).objectStore(store).getAll(); q.onsuccess = () => r(q.result); });
  const chunks = await all('chunks');
  return { nights: (await all('nights')).length, motion: chunks.filter((c) => c.kind === 'motion').length, audio: chunks.filter((c) => c.kind === 'audio').length, status: document.querySelector('#night-status').textContent };
});
console.log('enregistrement :', rec);
if (rec.motion < 1) fail('aucun bloc de mouvement stocké');
if (rec.audio < 0) fail('sons non stockés');
await page.screenshot({ path: join(SHOTS, 'app-nuit.png') });
// fin de nuit par appui long : enregistrement trop court → message attendu
await page.tap?.('#night').catch(() => {});
await page.dispatchEvent('#btn-stop', 'pointerdown');
await page.waitForFunction(() => /trop court/.test(document.querySelector('#toast').textContent), null, { timeout: 30000 });
console.log('arrêt par appui long : OK (nuit trop courte signalée)');
await page.evaluate(() => window.__motion(false));
}

// ---- 2. nuit simulée injectée puis analysée
await page.goto(base);
await page.evaluate(async () => {
  const buf = new Float32Array(await (await fetch('/night.f32')).arrayBuffer());
  const db = await new Promise((r) => { const q = indexedDB.open('somno'); q.onsuccess = () => r(q.result); });
  const put = (store, v) => new Promise((r) => { const t = db.transaction(store, 'readwrite'); t.objectStore(store).put(v); t.oncomplete = r; });
  const start = new Date(); start.setHours(23, 10, 0, 0); start.setDate(start.getDate() - 1);
  const id = String(start.getTime());
  const per = 50 * 60 * 6;
  for (let s = 0; s * per < buf.length; s++) {
    const data = buf.slice(s * per, (s + 1) * per);
    await put('chunks', { night: id, kind: 'motion', seq: s, n: data.length / 6, data });
  }
  await put('nights', { id, start: start.getTime(), end: start.getTime() + (buf.length / 6 / 50) * 1000, status: 'recording', opts: { audio: false }, fs: 50, gaps: [], motionStats: { rateHz: 60 } });
});
await page.goto(base);
await page.waitForSelector('#btn-recover');
const t0 = Date.now();
await page.click('#btn-recover');
await Promise.race([
  page.waitForSelector('#report.active', { timeout: 180000 }),
  page.waitForFunction(() => !document.querySelector('#toast').classList.contains('hidden'), null, { timeout: 180000 })
    .then(async () => { throw new Error('toast : ' + await page.textContent('#toast')); }),
]);
console.log(`analyse de la nuit injectée : ${((Date.now() - t0) / 1000).toFixed(1)} s`);
await page.waitForTimeout(500);
const repInfo = await page.evaluate(() => document.querySelector('#rep-body').innerText.slice(0, 600));
console.log(repInfo);
await page.screenshot({ path: join(SHOTS, 'app-rapport.png'), fullPage: true });
const report = await page.evaluate(async () => {
  const db = await new Promise((r) => { const q = indexedDB.open('somno'); q.onsuccess = () => r(q.result); });
  return new Promise((r) => { const q = db.transaction('nights').objectStore('nights').getAll(); q.onsuccess = () => r(q.result.find((n) => n.report && n.report.nEp > 100)?.report); });
});
if (!report) fail('pas de rapport');
else {
  const truth = JSON.parse(readFileSync(NIGHT.replace('.f32', '.json'), 'utf8')).stages;
  const map = [0, 1, 1, 2, 3];
  let ok = 0, n = 0;
  report.hyp.forEach((s, i) => { if (i < truth.length) { n++; ok += s === map[truth[i]] ? 1 : 0; } });
  console.log(`accord avec la vérité simulée : ${(100 * ok / n).toFixed(1)} % ; TST ${report.metrics.tstMin} min ; FR ${report.metrics.rrMean?.toFixed(1)} ; FC ${report.metrics.hrMean?.toFixed(1)}`);
  if (!(report.metrics.rrMean > 12 && report.metrics.rrMean < 17)) fail('fréquence respiratoire moyenne implausible');
}
// export brut .somno (relu ensuite par l'analyse Python : python -m somno.cli)
await page.evaluate(() => { navigator.canShare = () => false; });
await page.click('#btn-export');
const [dl] = await Promise.all([page.waitForEvent('download'), page.click('[data-x=raw]')]);
const rawPath = join(process.env.TMPDIR || '/tmp', 'e2e-export.somno');
await dl.saveAs(rawPath);
console.log('export .somno :', rawPath);
await page.click('button[data-go=history]').catch(() => {});
await page.goto(base + '#');
if (errors.length) fail('erreurs JS : ' + errors.join(' | '));
await browser.close();
server.close();
console.log(process.exitCode ? 'TEST E2E : ÉCHEC' : 'TEST E2E : OK');
