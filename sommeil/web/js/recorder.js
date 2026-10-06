// Enregistrement nocturne : accéléromètre + gyroscope (devicemotion, ré-échantillonnés à
// 50 Hz), sons (AudioWorklet, 4 caractéristiques/s), sonar optionnel, écran maintenu allumé.
import * as store from './storage.js';
import { SonarOnline } from './sonar.js';

export const FS = 50;
const CHUNK_S = 60;
const DEG = Math.PI / 180;

export class Recorder {
  constructor({ onLive, onStatus } = {}) {
    this.onLive = onLive || (() => {});
    this.onStatus = onStatus || (() => {});
    this.running = false;
    this.onMotion = this.onMotion.bind(this);
    this.onVisibility = this.onVisibility.bind(this);
  }

  // ------------------------------------------------------------------ démarrage
  async start(opts) {
    this.opts = opts;
    this.dry = !!opts.dry; // vérification du placement : rien n'est stocké
    const now = Date.now();
    this.night = {
      id: String(now), start: now, end: null, status: 'recording', opts,
      fs: FS, gaps: [], ua: navigator.userAgent, motionStats: null,
    };
    if (!this.dry) {
      await store.persist();
      await store.putNight(this.night);
    }
    this.seq = { motion: 0, audio: 0, sonar: 0 };
    this.buf = new Float32Array(FS * CHUNK_S * 6);
    this.nBuf = 0;
    this.nTotal = 0;
    this.prev = null;
    this.gridT = null;       // prochain instant de grille (ms, horloge de l'événement)
    this.tOrigin = null;
    this.stats = { events: 0, intervals: [], minStepAcc: Infinity, minStepGyro: Infinity, last: null };
    this.live = { motion: new Float32Array(FS * 600 * 6), n: 0, pos: 0 };
    this.running = true;
    window.addEventListener('devicemotion', this.onMotion);
    document.addEventListener('visibilitychange', this.onVisibility);
    if (!this.dry) await this.keepAwake();
    if (opts.audio || opts.sonar) {
      try { await this.startAudio(opts); } catch (e) { this.onStatus({ audioError: String(e) }); }
    }
    this.onStatus({ started: true, id: this.night.id });
    return this.night.id;
  }

  // ------------------------------------------------------------------ mouvement
  onMotion(ev) {
    if (!this.running) return;
    const a = ev.accelerationIncludingGravity, r = ev.rotationRate;
    if (!a || a.x == null) return;
    const t = ev.timeStamp;
    const s = [a.x, a.y, a.z, (r?.beta ?? 0) * DEG, (r?.gamma ?? 0) * DEG, (r?.alpha ?? 0) * DEG];
    const st = this.stats;
    st.events++;
    if (st.last) {
      st.intervals.length < 600 && st.intervals.push(t - st.last.t);
      for (let i = 0; i < 3; i++) {
        const d = Math.abs(s[i] - st.last.s[i]);
        if (d > 0) st.minStepAcc = Math.min(st.minStepAcc, d);
        const g = Math.abs(s[3 + i] - st.last.s[3 + i]);
        if (g > 0) st.minStepGyro = Math.min(st.minStepGyro, g);
      }
    }
    st.last = { t, s };
    if (this.prev === null) {
      this.prev = { t, s };
      this.gridT = t;
      this.tOrigin = t;
      this.night.motionT0 = Date.now();
    }
    const step = 1000 / FS;
    if (t - this.prev.t > 1000) {
      // trou (page en arrière-plan, appel…) : on le note et on comble avec la valeur courante
      this.night.gaps.push({ at: (this.prev.t - this.tOrigin) / 1000, dur: (t - this.prev.t) / 1000 });
      while (this.gridT <= t) { this.pushSample(s); this.gridT += step; }
    } else {
      while (this.gridT <= t) {
        const f = (this.gridT - this.prev.t) / Math.max(1e-6, t - this.prev.t);
        this.pushSample(this.prev.s.map((v, i) => v + (s[i] - v) * f));
        this.gridT += step;
      }
    }
    this.prev = { t, s };
  }

  pushSample(v) {
    this.buf.set(v, this.nBuf * 6);
    const L = this.live;
    const cap = L.motion.length / 6;
    L.motion.set(v, L.pos * 6);
    L.pos = (L.pos + 1) % cap;
    L.n = Math.min(cap, L.n + 1);
    this.nBuf++;
    this.nTotal++;
    if (this.nBuf === FS * CHUNK_S) this.flushMotion();
  }

  flushMotion() {
    if (!this.nBuf) return;
    const data = this.buf.slice(0, this.nBuf * 6);
    const chunk = { night: this.night.id, kind: 'motion', seq: this.seq.motion++, n: this.nBuf, data };
    this.nBuf = 0;
    if (!this.dry) store.putChunk(chunk).catch((e) => this.onStatus({ storeError: String(e) }));
    this.onLive({ seconds: this.nTotal / FS, live: this.live, stats: this.motionStats() });
  }

  /** Dernières secondes de mouvement, dans l'ordre : { acc: [x,y,z], gyro: [x,y,z] }. */
  liveSnapshot(seconds = 120) {
    const L = this.live;
    const cap = L.motion.length / 6;
    const n = Math.min(L.n, seconds * FS);
    const acc = [0, 1, 2].map(() => new Float64Array(n)), gyro = [0, 1, 2].map(() => new Float64Array(n));
    for (let k = 0; k < n; k++) {
      const i = ((L.pos - n + k) % cap + cap) % cap;
      for (let c = 0; c < 3; c++) { acc[c][k] = L.motion[i * 6 + c]; gyro[c][k] = L.motion[i * 6 + 3 + c]; }
    }
    return { acc, gyro, n };
  }

  motionStats() {
    const st = this.stats;
    const iv = st.intervals.slice().sort((a, b) => a - b);
    return {
      events: st.events,
      rateHz: iv.length ? 1000 / iv[iv.length >> 1] : 0,
      stepAcc: Number.isFinite(st.minStepAcc) ? st.minStepAcc : null,
      stepGyro: Number.isFinite(st.minStepGyro) ? st.minStepGyro : null,
    };
  }

  // ------------------------------------------------------------------ audio
  async startAudio(opts) {
    if (navigator.audioSession) {
      try { navigator.audioSession.type = opts.sonar ? 'play-and-record' : 'record'; } catch { /* ignoré */ }
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx({ sampleRate: 48000 });
    await this.ctx.audioWorklet.addModule(new URL('./audio-worklet.js', import.meta.url));
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'somno-audio', {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
      processorOptions: { sonar: !!opts.sonar, sonarGain: opts.sonarGain ?? 0.25 },
    });
    src.connect(this.node);
    if (opts.sonar) this.node.connect(this.ctx.destination);
    else {
      const mute = this.ctx.createGain();
      mute.gain.value = 0;
      this.node.connect(mute).connect(this.ctx.destination);
    }
    this.night.audioRate = this.ctx.sampleRate;
    this.audioBuf = [];
    this.audioT0 = null;
    this.bg = null;
    this.clips = 0;
    this.lastClipT = -1e9;
    if (opts.sonar) {
      this.sonar = new SonarOnline(this.ctx.sampleRate, (r) => {
        this.lastSonar = r;
        const data = new Float32Array(r.disp.length * 2);
        data.set(r.disp, 0);
        data.set(r.act, r.disp.length);
        if (!this.dry) store.putChunk({ night: this.night.id, kind: 'sonar', seq: r.epoch, n: r.disp.length, bin: r.bin, distM: r.distM, data });
        this.onLive({ sonar: { epoch: r.epoch, distM: r.distM } });
      });
    }
    this.node.port.onmessage = (e) => this.onAudio(e.data);
    if (this.ctx.state === 'suspended') await this.ctx.resume();
  }

  onAudio(m) {
    if (!this.running) return;
    if (m.type === 'feat') {
      if (this.audioT0 === null) { this.audioT0 = m.t; this.night.audioT0 = Date.now(); }
      this.audioBuf.push(...m.v);
      // bruit de fond glissant (lent) → détection d'événements sonores
      const db = m.v[0];
      this.bg = this.bg === null ? db : this.bg + (db < this.bg ? 0.05 : 0.002) * (db - this.bg);
      if (this.opts.clips !== false && db > this.bg + 15 && m.t - this.lastClipT > 30 && this.clips < 60) {
        this.lastClipT = m.t;
        this.clips++;
        const at = m.t - this.audioT0;
        setTimeout(() => this.node?.port.postMessage({ type: 'clip', id: at }), 5000); // 5 s de contexte après
      }
      if (this.audioBuf.length >= 4 * CHUNK_S * 8) this.flushAudio();
    } else if (m.type === 'sonar') {
      this.sonar?.push(m.re, m.im);
    } else if (m.type === 'clip') {
      if (!this.dry) store.putClip({ night: this.night.id, t: m.id, rate: m.rate, data: m.data });
    }
  }

  flushAudio() {
    if (!this.audioBuf?.length) return;
    const data = Float32Array.from(this.audioBuf);
    this.audioBuf = [];
    if (this.dry) return;
    store.putChunk({ night: this.night.id, kind: 'audio', seq: this.seq.audio++, n: data.length / 8, data });
  }

  // ------------------------------------------------------------------ écran allumé
  async keepAwake() {
    try {
      this.wakeLock = await navigator.wakeLock.request('screen');
      this.wakeLock.addEventListener?.('release', () => { if (this.running && document.visibilityState === 'visible') this.keepAwake(); });
      return;
    } catch { /* repli vidéo */ }
    if (!this.video) {
      const v = document.createElement('video');
      v.src = new URL('../assets/blank.mp4', import.meta.url).href;
      v.muted = true; v.loop = true; v.playsInline = true;
      v.setAttribute('playsinline', '');
      v.style.cssText = 'position:fixed;width:1px;height:1px;opacity:0.01;pointer-events:none;bottom:0;left:0';
      document.body.appendChild(v);
      this.video = v;
    }
    try { await this.video.play(); } catch { /* sera relancé au prochain geste */ }
  }

  onVisibility() {
    if (document.visibilityState === 'visible' && this.running) {
      this.keepAwake();
      if (this.ctx?.state === 'suspended') this.ctx.resume();
    } else if (this.running) {
      this.night.gaps.push({ hiddenAt: Date.now() });
    }
  }

  // ------------------------------------------------------------------ arrêt
  async stop() {
    if (!this.running) return null;
    this.running = false;
    window.removeEventListener('devicemotion', this.onMotion);
    document.removeEventListener('visibilitychange', this.onVisibility);
    this.flushMotion();
    this.sonar?.finish();
    this.flushAudio();
    try { await this.wakeLock?.release(); } catch { /* déjà libéré */ }
    this.video?.pause();
    this.stream?.getTracks().forEach((t) => t.stop());
    try { await this.ctx?.close(); } catch { /* déjà fermé */ }
    this.night.end = Date.now();
    this.night.status = 'recorded';
    this.night.motionStats = this.motionStats();
    this.night.samples = this.nTotal;
    if (!this.dry) await store.putNight(this.night);
    return this.night.id;
  }
}

/** Demande les autorisations capteurs (doit être appelé depuis un geste utilisateur). */
export async function requestMotionPermission() {
  if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
    return (await DeviceMotionEvent.requestPermission()) === 'granted';
  }
  return 'DeviceMotionEvent' in window;
}
