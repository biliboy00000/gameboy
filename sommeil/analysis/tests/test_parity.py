"""Parité numérique Python ⇄ JavaScript de toute la chaîne d'analyse iPhone.

On génère 40 min de signaux iPhone simulés, on les analyse en Python et avec
web/js/analysis.js (via Node), puis on compare toutes les caractéristiques par époque.

    python -m pytest tests/test_parity.py -q
"""
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
from somno import model as M  # noqa: E402
from somno import pipeline, simulate  # noqa: E402

WEB = os.path.abspath(os.path.join(HERE, "..", "..", "web"))

NODE_SCRIPT = r"""
import { readFileSync, writeFileSync } from 'node:fs';
import { phoneFeatures } from '%(web)s/js/analysis.js';
import { predictProba } from '%(web)s/js/model.js';
const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const f64 = (a) => Float64Array.from(a);
const acc = inp.acc.map(f64), gyro = inp.gyro.map(f64);
const r = phoneFeatures(acc, gyro, 50);
const feats = {};
for (const [k, v] of Object.entries(r.feats)) feats[k] = Array.from(v, (x) => (Number.isFinite(x) ? x : null));
let proba = null;
if (inp.model) {
  const X = Array.from({ length: r.nEp }, (_, i) => inp.model.features.map((n) => r.feats[n][i]));
  proba = predictProba(inp.model, X);
}
writeFileSync(process.argv[3], JSON.stringify({ feats, proba }));
"""


@pytest.fixture(scope="module")
def signals():
    nt = simulate.synthetic_night(hours=0.7, seed=3)
    acc, gyro, _ = simulate.synth_phone(nt["resp"], nt["beats"], nt["moving"], fs=50, seed=3,
                                        fs_resp=nt["fs_resp"], resp_tilt_deg=0.1, bcg_mg=1.0)
    return acc, gyro, nt


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_js_matches_python(signals):
    acc, gyro, nt = signals
    X, names = pipeline.phone_features(acc, gyro, 50.0)
    # petit modèle entraîné sur ces données pour vérifier aussi l'inférence JS
    y = M.to4(nt["stages"][: len(X)])
    clf = M.train_hgb(np.nan_to_num(X, nan=np.nan), y, max_iter=15, min_samples_leaf=5)
    mdl = M.export_hgb(clf, names)
    p_sk = clf.predict_proba(X)
    p_json = M.predict_json(mdl, X)
    np.testing.assert_allclose(p_json, p_sk, atol=1e-5)  # feuilles arrondies à 1e-6

    with tempfile.TemporaryDirectory() as tmp:
        inp, out, scr = (os.path.join(tmp, f) for f in ("in.json", "out.json", "run.mjs"))
        json.dump(dict(acc=acc.T.tolist(), gyro=gyro.T.tolist(), model=mdl), open(inp, "w"))
        open(scr, "w").write(NODE_SCRIPT % dict(web=WEB))
        subprocess.run(["node", scr, inp, out], check=True)
        res = json.load(open(out))
    js = res["feats"]
    assert set(js) == set(names), set(js) ^ set(names)
    worst = []
    for j, name in enumerate(names):
        a = X[:, j]
        b = np.array([np.nan if v is None else v for v in js[name]])
        assert np.array_equal(np.isfinite(a), np.isfinite(b)), name
        ok = np.isfinite(a)
        # tolérance mixte : les écarts-types glissants quasi nuls (annulation numérique) sont
        # comparés en absolu
        err = np.max(np.abs(a[ok] - b[ok]) / (1e-3 + np.abs(a[ok]))) if ok.any() else 0.0
        worst.append((err, name))
    worst.sort(reverse=True)
    assert worst[0][0] < 1e-4, worst[:5]  # FFT JS vs pocketfft : bruit ~1e-12 amplifié par les z-scores
    np.testing.assert_allclose(np.array(res["proba"]), p_json, atol=1e-6)
