// Inférence du classifieur de stades (arbres boostés exportés depuis Python) + lissage HMM.
// Format : voir analysis/somno/model.py (export_hgb).

function treeValue(nodes, x) {
  let i = 0;
  for (;;) {
    const nd = nodes[i];
    if (nd.length === 1) return nd[0];
    const v = x[nd[0]];
    if (v == null || Number.isNaN(v)) i = nd[4] ? nd[2] : nd[3];
    else i = v <= nd[1] ? nd[2] : nd[3];
  }
}

/** X : tableau de lignes (valeurs dans l'ordre model.features). → probabilités. */
export function predictProba(model, X) {
  const k = model.classes.length;
  return X.map((x) => {
    const raw = model.base.slice();
    for (const it of model.trees) for (let c = 0; c < it.length; c++) raw[c] += treeValue(it[c], x);
    if (k === 2 && raw.length === 1) {
      const p1 = 1 / (1 + Math.exp(-raw[0]));
      return [1 - p1, p1];
    }
    const m = Math.max(...raw);
    const e = raw.map((r) => Math.exp(r - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });
}

/** Lissage avant-arrière (HMM) : émissions ∝ p(classe|x) / p(classe). */
export function hmmSmooth(P, T, prior) {
  const n = P.length, k = P[0].length;
  const E = P.map((p) => {
    const e = p.map((v, i) => v / Math.max(prior[i], 1e-9));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  });
  const a = [E[0].map((v) => v / k)];
  const norm = (v) => { const s = v.reduce((x, y) => x + y, 0); return v.map((x) => x / s); };
  a[0] = norm(a[0]);
  for (let t = 1; t < n; t++) {
    const v = new Array(k).fill(0);
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let i = 0; i < k; i++) s += a[t - 1][i] * T[i][j];
      v[j] = s * E[t][j];
    }
    a.push(norm(v));
  }
  const b = new Array(n);
  b[n - 1] = new Array(k).fill(1);
  for (let t = n - 2; t >= 0; t--) {
    const v = new Array(k).fill(0);
    for (let i = 0; i < k; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += T[i][j] * E[t + 1][j] * b[t + 1][j];
      v[i] = s;
    }
    b[t] = norm(v);
  }
  return a.map((ai, t) => norm(ai.map((v, i) => v * b[t][i])));
}

export function argmax(p) {
  let b = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
  return b;
}
