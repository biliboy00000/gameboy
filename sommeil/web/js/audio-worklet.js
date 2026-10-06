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
