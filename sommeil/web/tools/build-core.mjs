// Construit web/core/somno-core.js : le traitement audio temps réel (caractéristiques
// sonores, extraits, sonar FMCW + démodulation en ligne) en un seul script classique,
// exécutable par JavaScriptCore dans l'app iOS native (sans modules ES ni AudioWorklet).
// Même code que l'app web → mêmes résultats (testé par analysis/tests/test_core_bundle.py).
//
//   node web/tools/build-core.mjs
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WEB = join(dirname(fileURLToPath(import.meta.url)), '..');
const strip = (src) => src
  .replace(/^import[^;]+;\s*$/gm, '')
  .replace(/^export (const|function|class|async function)/gm, '$1');

const dsp = strip(readFileSync(join(WEB, 'js/dsp.js'), 'utf8'));
const sonar = strip(readFileSync(join(WEB, 'js/sonar.js'), 'utf8'));
// l'AudioWorklet définit sa propre fft : on l'isole dans une portée
const worklet = readFileSync(join(WEB, 'js/audio-worklet.js'), 'utf8');

const out = `// GÉNÉRÉ par web/tools/build-core.mjs — ne pas modifier à la main.
var SomnoCore = (function () {
  'use strict';
  const __posted = [];
  var sampleRate = 48000;
  var currentTime = 0;
  class AudioWorkletProcessor { constructor() { this.port = { postMessage: (m) => __posted.push(m), onmessage: null }; } }
  let __Proc = null;
  function registerProcessor(name, cls) { __Proc = cls; }
${dsp}
${sonar}
  const __worklet = (function () {
${worklet}
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
`;
mkdirSync(join(WEB, 'core'), { recursive: true });
writeFileSync(join(WEB, 'core/somno-core.js'), out);
console.log('web/core/somno-core.js', out.length, 'octets');
