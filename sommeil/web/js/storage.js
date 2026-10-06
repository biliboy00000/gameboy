// Stockage local (IndexedDB) : rien ne quitte l'iPhone.
//   nights : métadonnées + rapport de chaque nuit
//   chunks : données brutes par blocs (mouvement 50 Hz, sons 4 Hz, sonar)
//   clips  : courts extraits sonores (ronflements, paroles…)
// Si IndexedDB est indisponible (navigation privée, WKWebView sur schéma personnalisé),
// repli sur un stockage en mémoire : l'analyse fonctionne, rien n'est conservé.

const DB_NAME = 'somno';
const DB_VERSION = 1;
let dbPromise = null;
export let persistent = true;

// ------------------------------------------------------------------ repli mémoire
const mem = { nights: new Map(), chunks: new Map(), clips: new Map() };
const key = (v) => JSON.stringify(v);
const memStore = (name) => ({
  put: (v) => {
    const k = name === 'nights' ? v.id : name === 'chunks' ? [v.night, v.kind, v.seq] : [v.night, v.t];
    mem[name].set(key(k), v);
  },
  all: () => [...mem[name].values()],
});

function open() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve) => {
    let req;
    try { req = indexedDB.open(DB_NAME, DB_VERSION); } catch { persistent = false; resolve(null); return; }
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('nights')) db.createObjectStore('nights', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('chunks')) {
        const s = db.createObjectStore('chunks', { keyPath: ['night', 'kind', 'seq'] });
        s.createIndex('night', 'night');
      }
      if (!db.objectStoreNames.contains('clips')) {
        const s = db.createObjectStore('clips', { keyPath: ['night', 't'] });
        s.createIndex('night', 'night');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => { persistent = false; resolve(null); };
  });
  return dbPromise;
}

function tx(store, mode, fn, memFn) {
  return open().then((db) => {
    if (!db) return memFn(memStore(store));
    return new Promise((resolve, reject) => {
      const t = db.transaction(store, mode);
      const s = t.objectStore(store);
      let out;
      Promise.resolve(fn(s)).then((v) => { out = v; });
      t.oncomplete = () => resolve(out);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  });
}

const reqP = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

export const putNight = (n) => tx('nights', 'readwrite', (s) => s.put(n), (m) => m.put(n));
export const getNight = (id) => tx('nights', 'readonly', (s) => reqP(s.get(id)), (m) => m.all().find((n) => n.id === id));
export const listNights = () => tx('nights', 'readonly', (s) => reqP(s.getAll()), (m) => m.all())
  .then((a) => a.sort((x, y) => y.start - x.start));

export const putChunk = (c) => tx('chunks', 'readwrite', (s) => s.put(c), (m) => m.put(c));
export const putClip = (c) => tx('clips', 'readwrite', (s) => s.put(c), (m) => m.put(c));

export function getChunks(night, kind) {
  return tx('chunks', 'readonly',
    (s) => reqP(s.getAll(IDBKeyRange.bound([night, kind, 0], [night, kind, Infinity]))),
    (m) => m.all().filter((c) => c.night === night && c.kind === kind).sort((a, b) => a.seq - b.seq));
}

export const getClips = (night) => tx('clips', 'readonly', (s) => reqP(s.index('night').getAll(night)),
  (m) => m.all().filter((c) => c.night === night));

export async function deleteNight(id) {
  const db = await open();
  if (!db) {
    for (const name of ['nights', 'chunks', 'clips']) {
      for (const [k, v] of mem[name]) if ((v.id ?? v.night) === id) mem[name].delete(k);
    }
    return;
  }
  await tx('nights', 'readwrite', (s) => s.delete(id));
  for (const store of ['chunks', 'clips']) {
    await tx(store, 'readwrite', (s) => new Promise((res) => {
      const r = s.index('night').openCursor(IDBKeyRange.only(id));
      r.onsuccess = () => { const c = r.result; if (c) { c.delete(); c.continue(); } else res(); };
    }));
  }
}

export async function persist() {
  try { return await navigator.storage?.persist?.(); } catch { return false; }
}

export async function usage() {
  try { return await navigator.storage?.estimate?.(); } catch { return null; }
}
