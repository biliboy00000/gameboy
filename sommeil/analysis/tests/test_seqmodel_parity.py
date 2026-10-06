"""Le GRU exporté en JSON donne les mêmes probabilités en PyTorch, NumPy et JavaScript."""
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
from somno import seqmodel as S  # noqa: E402

WEB = os.path.abspath(os.path.join(HERE, "..", "..", "web"))
torch = pytest.importorskip("torch")

NODE = r"""
import { readFileSync, writeFileSync } from 'node:fs';
import { predictSeq } from '%s/js/seqmodel.js';
const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const feats = {};
for (const [k, v] of Object.entries(inp.feats)) feats[k] = v.map((x) => (x === null ? NaN : x));
writeFileSync(process.argv[3], JSON.stringify(predictSeq(inp.model, feats, inp.n)));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_gru_export_matches():
    rng = np.random.default_rng(0)
    n = 120
    names = S.INPUTS + ["autre"]
    X = rng.normal(size=(n, len(names)))
    X[5:9, names.index("hr")] = np.nan
    Z, mu, sd = S.make_inputs(X, names)
    net = S.build_net(Z.shape[1], hidden=8, layers=2)
    net.eval()
    with torch.no_grad():
        lo = net(torch.tensor(Z[None], dtype=torch.float32))[0].numpy()
    p_torch = np.exp(lo - lo.max(1, keepdims=True))
    p_torch /= p_torch.sum(1, keepdims=True)
    m = S.export(net, mu, sd)
    p_np = S.predict_json(m, X, names)
    np.testing.assert_allclose(p_np, p_torch, atol=1e-5)
    with tempfile.TemporaryDirectory() as tmp:
        i, o, s = (os.path.join(tmp, f) for f in ("i.json", "o.json", "r.mjs"))
        feats = {c: [None if not np.isfinite(v) else float(v) for v in X[:, j]] for j, c in enumerate(names)}
        json.dump(dict(model=m, feats=feats, n=n), open(i, "w"))
        open(s, "w").write(NODE % WEB)
        subprocess.run(["node", s, i, o], check=True)
        p_js = np.array(json.load(open(o)))
    np.testing.assert_allclose(p_js, p_np, atol=1e-9)
