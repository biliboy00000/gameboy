"""Détection des ronflements : bouffées graves synthétiques au rythme respiratoire, passées
dans le vrai traitement audio (script de l'AudioWorklet) puis dans audioEpochs (night.js)."""
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import scipy.signal

HERE = os.path.dirname(__file__)
WEB = os.path.abspath(os.path.join(HERE, "..", "..", "web"))
NODE = r"""
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({});
vm.runInContext(fs.readFileSync('%(web)s/core/somno-core.js', 'utf8'), ctx);
const C = vm.runInContext('SomnoCore', ctx);
const x = Float32Array.from(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')));
C.init({ sonar: false, sampleRate: 16000 });
for (let s = 0; s < x.length; s += 1600) C.pushAudio(x.subarray(s, s + 1600));
const v = Float32Array.from(C.drain().feats);
import('%(web)s/js/night.js').then(({ audioEpochs }) => {
  const r = audioEpochs({ v, n: v.length / 8, rate: 4 }, Math.floor(v.length / 8 / 120), null);
  fs.writeFileSync(process.argv[3], JSON.stringify(r.epochs.map((e) => e && e.snore)));
});
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_snore_detection():
    subprocess.run(["node", os.path.join(WEB, "tools", "build-core.mjs")], check=True, capture_output=True)
    fs = 16000  # fréquence réduite pour garder le test rapide (les bandes < 8 kHz restent valides)
    rng = np.random.default_rng(0)
    dur = 300  # 10 époques : 5 calmes puis 5 de ronflement
    n = dur * fs
    x = 0.002 * rng.normal(size=n)  # fond de chambre
    b, a = scipy.signal.butter(2, [120, 600], btype="band", fs=fs)
    for start in np.arange(150, dur, 4.0):  # une bouffée par respiration (15 /min)
        i0, i1 = int(start * fs), int((start + 1.4) * fs)
        burst = scipy.signal.lfilter(b, a, rng.normal(size=i1 - i0)) * np.hanning(i1 - i0) * 0.2
        x[i0:i1] += burst
    with tempfile.TemporaryDirectory() as tmp:
        i, o, s = (os.path.join(tmp, f) for f in ("i.json", "o.json", "r.cjs"))
        json.dump(x.astype(np.float32).tolist(), open(i, "w"))
        open(s, "w").write(NODE % dict(web=WEB))
        subprocess.run(["node", s, i, o], check=True)
        snore = json.load(open(o))
    assert len(snore) == 10
    assert not any(snore[:5]), snore
    assert sum(snore[5:]) >= 4, snore
