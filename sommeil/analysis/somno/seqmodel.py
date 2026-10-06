"""Modèle séquentiel : GRU bidirectionnel sur la suite des époques d'une nuit.

Un scoreur humain regarde l'avant et l'après ; un GRU bidirectionnel fait de même sur
toute la nuit (contexte de plusieurs cycles de 90 min), là où les arbres ne voient que
des fenêtres glissantes de quelques minutes. Petit réseau (~40 k paramètres), exporté
en JSON et réévalué en JavaScript (web/js/seqmodel.js) sans bibliothèque.
"""
import json

import numpy as np

INPUTS = ["rr", "rr_z", "rr_spec_z", "ibi_cv", "ibi_cv_z", "amp_cv", "amp_cv_z", "reg", "reg_z",
          "spec_conc", "spec_conc_z", "amp_log", "n_breaths", "hr", "hr_z", "hr_std", "hr_std_z",
          "hr_q", "mov_frac", "mov_int", "t_since_mov", "t_to_mov", "t_frac", "t_hours"]
MISSING = ["rr", "hr"]  # indicateurs de valeur manquante


def make_inputs(X, names, mean=None, std=None):
    """(n_ep, k) caractéristiques → (n_ep, d) entrées normalisées sans NaN."""
    cols = [X[:, names.index(c)] for c in INPUTS]
    Z = np.stack(cols, axis=1).astype(np.float64)
    miss = np.stack([~np.isfinite(X[:, names.index(c)]) for c in MISSING], axis=1).astype(np.float64)
    if mean is None:
        mean = np.nan_to_num(np.nanmean(Z, axis=0), nan=0.0)
        std = np.nan_to_num(np.nanstd(Z, axis=0), nan=1.0) + 1e-6
    Z = (Z - mean) / std
    Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -6, 6)
    return np.concatenate([Z, miss], axis=1), mean, std


def build_net(d_in, hidden=48, layers=2, n_out=4, dropout=0.2):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d_in, hidden)
            self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                              bidirectional=True, dropout=dropout if layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(2 * hidden, n_out)

        def forward(self, x):
            import torch
            h = torch.relu(self.inp(x))
            h, _ = self.gru(self.drop(h))
            return self.out(self.drop(h))

    return Net()


def train(seqs, labels, epochs=40, hidden=48, layers=2, lr=2e-3, seed=0, verbose=False,
          crop=(480, 1100), noise=0.1):
    """seqs : liste de (n_i, d) ; labels : liste de (n_i,) avec -1 = non scoré."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(4)
    d = seqs[0].shape[1]
    net = build_net(d, hidden, layers)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(seqs) + 15) // 16))
    w = np.bincount(np.concatenate([l[l >= 0] for l in labels]), minlength=4).astype(float)
    cw = torch.tensor((w.sum() / (4 * w)) ** 0.5, dtype=torch.float32)  # pondération douce
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(seqs))
        tot = 0.0
        for b in range(0, len(order), 16):
            batch = order[b:b + 16]
            L = int(rng.integers(crop[0], crop[1]))
            xs, ys = [], []
            for i in batch:
                n = len(seqs[i])
                s = int(rng.integers(0, max(1, n - L + 1)))
                x = seqs[i][s:s + L]
                y = labels[i][s:s + L]
                pad = L - len(x)
                if pad > 0:
                    x = np.concatenate([x, np.zeros((pad, d))])
                    y = np.concatenate([y, -np.ones(pad, dtype=int)])
                xs.append(x)
                ys.append(y)
            xt = torch.tensor(np.stack(xs), dtype=torch.float32)
            xt = xt + noise * torch.randn_like(xt)
            yt = torch.tensor(np.stack(ys), dtype=torch.long)
            logits = net(xt)
            loss = F.cross_entropy(logits.reshape(-1, 4), yt.reshape(-1), weight=cw, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        if verbose:
            print(f"  époque {ep + 1}/{epochs} perte {tot / max(1, len(order) // 16):.3f}", flush=True)
    net.eval()
    return net


def predict(net, seq):
    import torch
    with torch.no_grad():
        logits = net(torch.tensor(seq[None], dtype=torch.float32))[0].numpy()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def export(net, mean, std, names_in=INPUTS, extra=None):
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    r = lambda a: np.round(a.astype(np.float64), 7).tolist()
    layers = []
    li = 0
    while f"gru.weight_ih_l{li}" in sd:
        layer = {}
        for dirn, suf in (("fwd", ""), ("bwd", "_reverse")):
            layer[dirn] = dict(Wi=r(sd[f"gru.weight_ih_l{li}{suf}"]), Wh=r(sd[f"gru.weight_hh_l{li}{suf}"]),
                               bi=r(sd[f"gru.bias_ih_l{li}{suf}"]), bh=r(sd[f"gru.bias_hh_l{li}{suf}"]))
        layers.append(layer)
        li += 1
    d = dict(kind="bigru", inputs=list(names_in), missing=list(MISSING), mean=r(mean), std=r(std),
             inp=dict(W=r(sd["inp.weight"]), b=r(sd["inp.bias"])), gru=layers,
             out=dict(W=r(sd["out.weight"]), b=r(sd["out.bias"])))
    if extra:
        d.update(extra)
    return d


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _gru_dir(x, p, reverse):
    Wi, Wh, bi, bh = (np.asarray(p[k]) for k in ("Wi", "Wh", "bi", "bh"))
    H = Wh.shape[1]
    n = len(x)
    out = np.zeros((n, H))
    h = np.zeros(H)
    rng = range(n - 1, -1, -1) if reverse else range(n)
    gi_all = x @ Wi.T + bi
    for t in rng:
        gi = gi_all[t]
        gh = Wh @ h + bh
        r = _sig(gi[:H] + gh[:H])
        z = _sig(gi[H:2 * H] + gh[H:2 * H])
        nn_ = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        h = (1 - z) * nn_ + z * h
        out[t] = h
    return out


def predict_json(m, X, names):
    """Référence NumPy de l'inférence (pour tester le JS)."""
    Z, _, _ = make_inputs(X, names, np.asarray(m["mean"]), np.asarray(m["std"]))
    h = np.maximum(0, Z @ np.asarray(m["inp"]["W"]).T + np.asarray(m["inp"]["b"]))
    for layer in m["gru"]:
        h = np.concatenate([_gru_dir(h, layer["fwd"], False), _gru_dir(h, layer["bwd"], True)], axis=1)
    lo = h @ np.asarray(m["out"]["W"]).T + np.asarray(m["out"]["b"])
    e = np.exp(lo - lo.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def save(m, path):
    with open(path, "w") as f:
        json.dump(m, f, separators=(",", ":"), allow_nan=False)
