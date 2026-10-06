// Exports (rapport JSON, époques CSV, données brutes .somno) et lecture des extraits sonores.
import * as store from './storage.js';

const STAGES = ['Éveil', 'Léger', 'Profond', 'REM'];

async function save(blob, filename) {
  const file = new File([blob], filename, { type: blob.type });
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: filename }); return; } catch (e) { if (e.name === 'AbortError') return; }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 4000);
}

const stamp = (t) => new Date(t).toISOString().slice(0, 16).replace(/[:T]/g, '-');

export async function exportJson(rep) {
  await save(new Blob([JSON.stringify(rep, null, 1)], { type: 'application/json' }), `somno-${stamp(rep.start)}.json`);
}

export async function exportCsv(rep) {
  const rows = ['epoque;heure;stade;p_eveil;p_leger;p_profond;p_rem;resp_min;fc_bpm;mouvement;ronflement;niveau_db'];
  for (let i = 0; i < rep.nEp; i++) {
    const t = new Date(rep.start + i * 30e3).toLocaleTimeString('fr-FR');
    const p = rep.proba[i];
    const v = (x) => (x == null ? '' : String(x).replace('.', ','));
    rows.push([i, t, STAGES[rep.hyp[i]], ...p.map(v), v(rep.series.rr[i]), v(rep.series.hr[i]), v(rep.series.mov[i]),
      rep.series.snore ? rep.series.snore[i] : '', rep.series.db ? v(rep.series.db[i]) : ''].join(';'));
  }
  await save(new Blob(['﻿' + rows.join('\n')], { type: 'text/csv' }), `somno-${stamp(rep.start)}.csv`);
}

/** Format .somno : "SOMNO001" | uint32 longueur d'en-tête | en-tête JSON | sections Float32 LE.
 *  Lisible par analysis/somno/io.py (read_somno). */
export async function exportRaw(id) {
  const night = await store.getNight(id);
  const sections = [];
  const blobs = [];
  let offset = 0;
  for (const kind of ['motion', 'audio', 'sonar']) {
    const chunks = (await store.getChunks(id, kind)).sort((a, b) => a.seq - b.seq);
    if (!chunks.length) continue;
    const cols = kind === 'motion' ? 6 : kind === 'audio' ? 8 : 2;
    let n = 0;
    const parts = [];
    for (const c of chunks) {
      if (kind === 'sonar') {
        // [déplacement, activité] entrelacés, une ligne par trame
        const inter = new Float32Array(c.n * 2);
        for (let i = 0; i < c.n; i++) { inter[2 * i] = c.data[i]; inter[2 * i + 1] = c.data[c.n + i]; }
        parts.push(inter);
      } else parts.push(c.data);
      n += c.n;
    }
    const bytes = n * cols * 4;
    sections.push({
      kind, n, cols, dtype: 'float32', offset, bytes,
      columns: kind === 'motion' ? ['ax', 'ay', 'az', 'gx', 'gy', 'gz'] : kind === 'audio'
        ? ['db', 'b50_150', 'b150_500', 'b500_1k', 'b1k_2k', 'b2k_4k', 'b4k_8k', 'peak_db'] : ['disp_mm', 'act'],
      rate: kind === 'motion' ? night.fs : kind === 'audio' ? 4 : (night.audioRate || 48000) / 1024,
      epochs: kind === 'sonar' ? chunks.map((c) => ({ seq: c.seq, bin: c.bin, distM: c.distM })) : undefined,
    });
    blobs.push(...parts.map((p) => p.buffer.slice(p.byteOffset, p.byteOffset + p.byteLength)));
    offset += bytes;
  }
  const meta = { ...night, report: undefined, sections, format: 'somno/1' };
  const head = new TextEncoder().encode(JSON.stringify(meta));
  const pre = new ArrayBuffer(12);
  new Uint8Array(pre, 0, 8).set(new TextEncoder().encode('SOMNO001'));
  new DataView(pre).setUint32(8, head.length, true);
  await save(new Blob([pre, head, ...blobs], { type: 'application/octet-stream' }), `somno-${stamp(night.start)}.somno`);
}

let audioCtx = null;
export async function playClip(night, t) {
  const clips = await store.getClips(night);
  const c = clips.find((x) => x.t === t);
  if (!c) return;
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  const buf = audioCtx.createBuffer(1, c.data.length, c.rate);
  // normalisation douce pour entendre les bruits faibles
  let mx = 1e-6;
  for (const v of c.data) mx = Math.max(mx, Math.abs(v));
  const g = Math.min(20, 0.8 / mx);
  buf.getChannelData(0).set(c.data.map((v) => v * g));
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  src.start();
}
