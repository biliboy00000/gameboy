// Service worker : l'application fonctionne hors ligne (la nuit, sans réseau).
const CACHE = 'somno-v2';
const FILES = [
  './', 'index.html', 'css/app.css', 'manifest.webmanifest', 'assets/blank.mp4', 'assets/icon-180.png',
  'model/staging.json', 'model/apnea.json',
  'js/app.js', 'js/analysis.js', 'js/apnea.js', 'js/audio-worklet.js', 'js/charts.js', 'js/dsp.js',
  'js/export.js', 'js/import.js', 'js/model.js', 'js/night.js', 'js/recorder.js', 'js/seqmodel.js', 'js/sonar.js', 'js/storage.js',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// réseau d'abord (mises à jour), cache en secours (hors ligne)
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request).then((r) => {
    const copy = r.clone();
    caches.open(CACHE).then((c) => c.put(e.request, copy));
    return r;
  }).catch(() => caches.match(e.request)));
});
