// Inférence du GRU bidirectionnel exporté par analysis/somno/seqmodel.py.

const sig = (x) => 1 / (1 + Math.exp(-x));

function matVec(W, x, b) {
  const out = new Float64Array(W.length);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let s = b ? b[i] : 0;
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

function gruDir(xs, p, reverse) {
  const H = p.Wh[0].length;
  const n = xs.length;
  const out = new Array(n);
  let h = new Float64Array(H);
  for (let k = 0; k < n; k++) {
    const t = reverse ? n - 1 - k : k;
    const gi = matVec(p.Wi, xs[t], p.bi);
    const gh = matVec(p.Wh, h, p.bh);
    const nh = new Float64Array(H);
    for (let i = 0; i < H; i++) {
      const r = sig(gi[i] + gh[i]);
      const z = sig(gi[H + i] + gh[H + i]);
      const nn = Math.tanh(gi[2 * H + i] + r * gh[2 * H + i]);
      nh[i] = (1 - z) * nn + z * h[i];
    }
    h = nh;
    out[t] = h;
  }
  return out;
}

/** feats : dictionnaire nom → tableau par époque. → probabilités (n, 4). */
export function predictSeq(m, feats, nEp) {
  const xs = [];
  for (let t = 0; t < nEp; t++) {
    const z = m.inputs.map((name, j) => {
      const v = feats[name]?.[t];
      if (v == null || !Number.isFinite(v)) return 0;
      return Math.max(-6, Math.min(6, (v - m.mean[j]) / m.std[j]));
    });
    for (const name of m.missing) z.push(Number.isFinite(feats[name]?.[t]) ? 0 : 1);
    xs.push(z);
  }
  let h = xs.map((x) => matVec(m.inp.W, x, m.inp.b).map((v) => Math.max(0, v)));
  for (const layer of m.gru) {
    const f = gruDir(h, layer.fwd, false), b = gruDir(h, layer.bwd, true);
    h = f.map((v, t) => Float64Array.from([...v, ...b[t]]));
  }
  return h.map((x) => {
    const lo = matVec(m.out.W, x, m.out.b);
    const mx = Math.max(...lo);
    const e = Array.from(lo, (v) => Math.exp(v - mx));
    const s = e.reduce((a, c) => a + c, 0);
    return e.map((v) => v / s);
  });
}
