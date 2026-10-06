"""Le détecteur d'événements respiratoires donne les mêmes événements en Python et en JS."""
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
from somno import apnea  # noqa: E402

WEB = os.path.abspath(os.path.join(HERE, "..", "..", "web"))
NODE = r"""
import { readFileSync, writeFileSync } from 'node:fs';
import { detectEvents, rdi } from '%s/js/apnea.js';
const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const ev = detectEvents(Float64Array.from(inp.w), inp.moving);
writeFileSync(process.argv[3], JSON.stringify({ ev, rdi: rdi(ev, inp.sleep) }));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_apnea_js_matches_python():
    rng = np.random.default_rng(1)
    fs = 10
    t = np.arange(0, 3600, 1 / fs)
    amp = np.ones_like(t)
    for s in rng.uniform(100, 3400, 25):  # 25 chutes d'amplitude de 15–40 s
        amp[(t > s) & (t < s + rng.uniform(15, 40))] = rng.uniform(0.05, 0.4)
    w = amp * np.sin(2 * np.pi * 0.25 * t) + 0.02 * rng.normal(size=len(t))
    moving = np.zeros(3600, dtype=bool)
    moving[1800:1805] = True
    sleep = np.ones(120, dtype=bool)
    ev = apnea.detect_events(w, moving)
    r, n = apnea.rdi(ev, sleep)
    assert 15 <= len(ev) <= 30
    with tempfile.TemporaryDirectory() as tmp:
        i, o, s = (os.path.join(tmp, f) for f in ("i.json", "o.json", "r.mjs"))
        json.dump(dict(w=w.tolist(), moving=moving.tolist(), sleep=sleep.tolist()), open(i, "w"))
        open(s, "w").write(NODE % WEB)
        subprocess.run(["node", s, i, o], check=True)
        res = json.load(open(o))
    assert len(res["ev"]) == len(ev)
    np.testing.assert_allclose(np.array(res["ev"]), np.array(ev), atol=1e-9)
    assert abs(res["rdi"]["rdi"] - r) < 1e-9
