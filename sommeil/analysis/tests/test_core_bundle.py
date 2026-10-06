"""Le script autonome web/core/somno-core.js (exécuté par JavaScriptCore dans l'app iOS)
donne les mêmes époques sonar que somno.sonar, et des caractéristiques sonores."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from somno import sonar  # noqa: E402

WEB = os.path.abspath(os.path.join(HERE, "..", "..", "web"))

NODE = r"""
const fs = require('fs');
const vm = require('vm');
// exécution comme un script classique, sans module (comme JavaScriptCore)
const ctx = vm.createContext({});
vm.runInContext(fs.readFileSync('%s/core/somno-core.js', 'utf8'), ctx);
const SomnoCore = vm.runInContext('SomnoCore', ctx);
const x = Float32Array.from(JSON.parse(fs.readFileSync(process.argv[2], 'utf8')));
SomnoCore.init({ sonar: true, sampleRate: 48000 });
const epochs = [];
let feats = [];
for (let s = 0; s < x.length; s += 4800) {   // blocs de 100 ms comme l'app iOS
  SomnoCore.pushAudio(x.subarray(s, s + 4800));
  const r = SomnoCore.drain(); epochs.push(...r.epochs); feats.push(...r.feats);
}
SomnoCore.finish();
epochs.push(...SomnoCore.drain().epochs);
fs.writeFileSync(process.argv[3], JSON.stringify({ epochs, nfeat: feats.length }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_core_bundle_sonar():
    subprocess.run(["node", os.path.join(WEB, "tools", "build-core.mjs")], check=True)
    fsf = sonar.FRAME_RATE
    k = int(100 * fsf)
    t = np.arange(k) / fsf
    frames, tx = sonar.simulate_frames(3.0 * np.sin(2 * np.pi * 14 / 60 * t), 1.0, k, seed=5, noise_db=-70)
    disp, bins, act = sonar.displacement(sonar.align_direct(sonar.range_profiles(frames, tx)))
    with tempfile.TemporaryDirectory() as tmp:
        i, o, s = (os.path.join(tmp, f) for f in ("i.json", "o.json", "r.js"))
        json.dump(frames.ravel().astype(np.float32).tolist(), open(i, "w"))
        open(s, "w").write(NODE % WEB)
        subprocess.run(["node", s, i, o], check=True)
        res = json.load(open(o))
    ne = int(round(30 * fsf))
    assert len(res["epochs"]) == len(bins) == 3
    for ep in res["epochs"]:
        e = ep["epoch"]
        assert ep["bin"] == bins[e]
        np.testing.assert_allclose(ep["disp"], disp[e * ne:(e + 1) * ne], atol=0.02)
    # 100 s de son → 400 trames × 8 caractéristiques
    assert abs(res["nfeat"] - 400 * 8) <= 16
