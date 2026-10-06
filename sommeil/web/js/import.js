// Import d'un fichier .somno (export de l'app web ou de l'app iOS native) dans IndexedDB.
import * as store from './storage.js';

export async function importSomno(buffer) {
  const u8 = new Uint8Array(buffer);
  const magic = new TextDecoder().decode(u8.subarray(0, 8));
  if (magic !== 'SOMNO001') throw new Error('Fichier .somno invalide');
  const hlen = new DataView(buffer).getUint32(8, true);
  const meta = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + hlen)));
  const dataStart = 12 + hlen;
  const id = String(meta.id || meta.start || Date.now());
  const night = { ...meta, id, status: 'recorded', imported: true };
  delete night.sections;
  for (const s of meta.sections) {
    // copie alignée sur 4 octets
    const raw = buffer.slice(dataStart + s.offset, dataStart + s.offset + s.bytes);
    const all = new Float32Array(raw);
    if (s.kind === 'motion' || s.kind === 'audio') {
      const per = s.kind === 'motion' ? 50 * 60 : 4 * 60; // un bloc par minute
      for (let k = 0, seq = 0; k < s.n; k += per, seq++) {
        const n = Math.min(per, s.n - k);
        await store.putChunk({ night: id, kind: s.kind, seq, n, data: all.slice(k * s.cols, (k + n) * s.cols) });
      }
      if (s.kind === 'motion') night.fs = s.rate;
    } else if (s.kind === 'sonar') {
      const ne = s.epochs.length ? Math.round(s.n / s.epochs.length) : 0;
      s.epochs.forEach((ep, i) => {
        const data = new Float32Array(2 * ne);
        for (let t = 0; t < ne; t++) { data[t] = all[(i * ne + t) * 2]; data[ne + t] = all[(i * ne + t) * 2 + 1]; }
        store.putChunk({ night: id, kind: 'sonar', seq: ep.seq, n: ne, bin: ep.bin, distM: ep.distM, data });
      });
      night.audioRate = Math.round(s.rate * 1024);
    } else if (s.kind === 'depth') {
      // profondeur TrueDepth (app iOS) : traitée comme une onde de déplacement (mm)
      await store.putChunk({ night: id, kind: 'depth', seq: 0, n: s.n, rate: s.rate, data: all });
    }
  }
  await store.putNight(night);
  return id;
}
