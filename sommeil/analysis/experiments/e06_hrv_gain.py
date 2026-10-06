"""Expérience 6 — La variabilité cardiaque battement par battement (VFC) vaut-elle l'effort ?

Si l'iPhone pouvait mesurer chaque intervalle entre battements (et pas seulement la FC
moyenne toutes les 5 s), gagnerait-on en précision des stades ? On calcule la VFC à partir
des vrais pics R de l'ECG (borne haute) — RMSSD, SDNN, pNN50, puissances LF/HF — et on
compare la validation croisée avec / sans ces caractéristiques.

    python experiments/e06_hrv_gain.py
"""
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import dsp, features, heart  # noqa: E402
from somno import model as M  # noqa: E402

HRV = ["rmssd", "sdnn", "pnn50", "lf", "hf", "lfhf"]


def hrv_epochs(rp, n_ep, win_s=300):
    rr = np.diff(rp)
    t = rp[1:]
    ok = (rr > 0.33) & (rr < 2.0)
    # rejet des battements ectopiques / ratés : écart > 20 % à la médiane locale
    med = np.array([np.median(rr[max(0, i - 5): i + 6]) for i in range(len(rr))])
    ok &= np.abs(rr - med) < 0.2 * med
    F = np.full((n_ep, len(HRV)), np.nan)
    for e in range(n_ep):
        c = (e + 0.5) * 30
        sel = ok & (np.abs(t - c) <= win_s / 2)
        x, tt = rr[sel], t[sel]
        if len(x) < 60:
            continue
        d = np.diff(x)
        F[e, 0] = np.log(np.sqrt(np.mean(d * d)) * 1000 + 1e-3)
        F[e, 1] = np.log(np.std(x) * 1000 + 1e-3)
        F[e, 2] = np.mean(np.abs(d) > 0.05)
        # spectre : RR ré-échantillonné à 4 Hz
        g = np.arange(tt[0], tt[-1], 0.25)
        if len(g) < 256:
            continue
        y = np.interp(g, tt, x)
        f, p = dsp.hann_spectrum(y, 4.0, 1024)
        lf = p[(f >= 0.04) & (f < 0.15)].sum()
        hf = p[(f >= 0.15) & (f < 0.4)].sum()
        F[e, 3] = np.log(lf + 1e-12)
        F[e, 4] = np.log(hf + 1e-12)
        F[e, 5] = np.log((lf + 1e-12) / (hf + 1e-12))
    return F


def one(path):
    night = dict(np.load(path))
    rp = heart.detect_r_peaks(night["ecg"], 200.0) / 200.0
    F = hrv_epochs(rp, len(night["stages"]))
    # normalisation par nuit + contexte (comme les autres caractéristiques)
    cols = []
    for j in range(F.shape[1]):
        v = F[:, j]
        z = (v - np.nanmedian(v)) / dsp.robust_scale(v) if np.isfinite(v).any() else v
        cols += [v, z, features.rolling(z, 11, "mean")]
    return os.path.basename(path)[:-4], np.stack(cols, axis=1).astype(np.float32)


def main():
    d = np.load("/home/user/data/feats_c18.npz")
    rids = [str(r) for r in d["rids"]]
    files = [f"/home/user/data/c18/{r}.npz" for r in rids]
    with ProcessPoolExecutor(2) as ex:
        res = dict(ex.map(one, files))
    night = d["night"]
    H = np.concatenate([res[r][: (night == i).sum()] for i, r in enumerate(rids)])
    names = [str(n) for n in d["names"]]
    y4 = M.to4(d["y"])
    Xp = d["Xp"].astype(float)
    cols = [names.index(n) for n in names if n != "amp"]
    ids = np.unique(night)
    rng = np.random.default_rng(0)
    rng.shuffle(ids)
    out = {}
    for label, X in (("PSG sans VFC", Xp[:, cols]), ("PSG + VFC battement par battement", np.concatenate([Xp[:, cols], H], 1))):
        P = np.zeros((len(y4), 4))
        for k in range(5):
            test = np.isin(night, ids[k::5])
            tr = (~test) & (y4 >= 0)
            clf = M.train_hgb(X[tr], y4[tr], max_iter=200, class_weight=None)
            P[test] = clf.predict_proba(X[test])
        k4, _ = M.cohen_kappa(y4, P.argmax(1), 4)
        ok = y4 >= 0
        out[label] = dict(kappa4=float(k4), acc4=float((P.argmax(1)[ok] == y4[ok]).mean()))
        print(label, out[label], flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "e06_hrv.json"), "w"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()
