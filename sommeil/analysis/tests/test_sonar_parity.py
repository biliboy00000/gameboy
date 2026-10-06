"""Parité Python ⇄ JS de la chaîne sonar : AudioWorklet (profils de distance) puis
traitement en ligne (web/js/sonar.js) contre somno.sonar, sur 3 min simulées."""
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
import { readFileSync, writeFileSync } from 'node:fs';
const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
// --- mini-environnement AudioWorklet
let Proc = null;
globalThis.sampleRate = 48000;
globalThis.currentTime = 0;
globalThis.AudioWorkletProcessor = class { constructor() { this.port = { postMessage: (m) => posted.push(m), onmessage: null }; } };
globalThis.registerProcessor = (name, cls) => { Proc = cls; };
const posted = [];
await import('%(web)s/js/audio-worklet.js');
const p = new Proc({ processorOptions: { sonar: true, sonarGain: 0.25 } });
const x = Float32Array.from(inp.signal);
const out = [new Float32Array(128)];
for (let s = 0; s + 128 <= x.length; s += 128) p.process([[x.subarray(s, s + 128)]], [out]);
const profs = posted.filter((m) => m.type === 'sonar');
const { SonarOnline } = await import('%(web)s/js/sonar.js');
const epochs = [];
const so = new SonarOnline(48000, (r) => epochs.push({ e: r.epoch, bin: r.bin, disp: Array.from(r.disp), act: Array.from(r.act) }));
for (const m of profs) so.push(m.re, m.im);
so.finish();
writeFileSync(process.argv[3], JSON.stringify({
  prof0: { re: Array.from(profs[5].re), im: Array.from(profs[5].im) }, nprof: profs.length, epochs,
  chirp: Array.from(p.chirp),
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_sonar_js_matches_python():
    fsf = sonar.FRAME_RATE
    k = int(150 * fsf)
    t = np.arange(k) / fsf
    chest = 3.0 * np.sin(2 * np.pi * 15 / 60 * t)
    frames, tx = sonar.simulate_frames(chest, 0.8, k, seed=2, noise_db=-70)
    sig = frames.ravel().astype(np.float32)
    prof = sonar.range_profiles(frames, tx)
    disp, bins, act = sonar.displacement(sonar.align_direct(prof))
    with tempfile.TemporaryDirectory() as tmp:
        i, o, s = (os.path.join(tmp, f) for f in ("i.json", "o.json", "r.mjs"))
        json.dump(dict(signal=sig.tolist()), open(i, "w"))
        open(s, "w").write(NODE % dict(web=WEB))
        subprocess.run(["node", s, i, o], check=True)
        res = json.load(open(o))
    np.testing.assert_allclose(res["chirp"], tx, atol=1e-6)
    assert res["nprof"] == k
    p5 = np.array(res["prof0"]["re"]) + 1j * np.array(res["prof0"]["im"])
    np.testing.assert_allclose(p5, prof[5], atol=1e-4 * np.abs(prof[5]).max())
    ne = int(round(30 * fsf))
    assert len(res["epochs"]) == len(bins)
    for ep in res["epochs"]:
        e = ep["e"]
        assert ep["bin"] == bins[e]
        np.testing.assert_allclose(ep["disp"], disp[e * ne:(e + 1) * ne], atol=0.02)  # mm
        a_js, a_py = np.array(ep["act"]), act[e * ne:(e + 1) * ne]
        np.testing.assert_allclose(a_js[1:], a_py[1:], rtol=1e-3, atol=1e-3 * a_py.max())
