"""Expérience 4 — Dépistage des apnées du sommeil à partir de la respiration.

Compare l'indice d'événements respiratoires estimé (IDR) à l'IAH de la PSG, nuit par
nuit, (a) depuis les ceintures PSG (borne haute) et (b) depuis la chaîne iPhone
simulée (hybride). On mesure la corrélation et la capacité à repérer un SAOS
modéré-sévère (IAH ≥ 15) ou sévère (IAH ≥ 30).

    python experiments/e04_apnea.py --nights 120 --hybrid 60
"""
import argparse
import glob
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import apnea, heart, movement, pipeline, respiration  # noqa: E402

DROPS = [0.3, 0.4, 0.5]


def psg_ahi(night):
    tst_h = (night["stages"] > 0).sum() * 30 / 3600
    keep = np.isin(night["ev_type"], ["obstructiveapnea", "centralapnea", "mixedapnea", "hypopnea"])
    return keep.sum() / max(tst_h, 0.5)


def one(job):
    path, hybrid = job
    night = dict(np.load(path))
    sleep = night["stages"] > 0
    out = dict(rid=os.path.basename(path)[:-4], ahi=float(psg_ahi(night)))
    belts = np.stack([night["chest"], night["abd"]], axis=1)
    rc = respiration.belts_to_resp_channels(belts, float(night["fs_resp"]))
    w, _ = respiration.pca_waveform(rc)
    mv = pipeline.psg_movement(night) > movement.THR
    for d in DROPS:
        out[f"psg_{d}"] = apnea.rdi(apnea.detect_events(w, mv, drop=d), sleep)[0]
    if hybrid:
        rid = out["rid"]
        seed = zlib.crc32(rid.encode())
        rng = np.random.default_rng(seed)
        rp = heart.detect_r_peaks(night["ecg"], 200.0) / 200.0
        kw = dict(resp_tilt_deg=float(np.exp(rng.uniform(np.log(0.03), np.log(0.5)))),
                  bcg_mg=float(np.exp(rng.uniform(np.log(0.2), np.log(3.0)))))
        acc, gyro, _ = pipeline.hybrid_signals(night, rp, seed=seed % 10000, **kw)
        rc = respiration.motion_to_resp_channels(acc, gyro, 50.0)
        w2, _ = respiration.pca_waveform(rc)
        act = movement.relative_activity(movement.activity_per_second(acc, 50.0),
                                         movement.activity_per_second(gyro, 50.0))
        for d in DROPS:
            out[f"hyb_{d}"] = apnea.rdi(apnea.detect_events(w2, act > movement.THR, drop=d), sleep)[0]
        out["tilt"] = kw["resp_tilt_deg"]
    return out


def summarize(res, key):
    ahi = np.array([r["ahi"] for r in res if key in r])
    est = np.array([r[key] for r in res if key in r])
    if len(ahi) < 5:
        return None
    rho = float(np.corrcoef(ahi, est)[0, 1])
    from scipy.stats import spearmanr
    sp = float(spearmanr(ahi, est).statistic)
    s = dict(n=len(ahi), pearson=rho, spearman=sp)
    for thr in (15, 30):
        pos = ahi >= thr
        # seuil de l'estimateur choisi pour maximiser l'indice de Youden (optimiste) et
        # rapporté avec sa sensibilité/spécificité
        best = (0, None)
        for t in np.unique(est):
            se = np.mean(est[pos] >= t) if pos.any() else np.nan
            sp_ = np.mean(est[~pos] < t) if (~pos).any() else np.nan
            if se + sp_ - 1 > best[0]:
                best = (se + sp_ - 1, (float(t), float(se), float(sp_)))
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(pos, est)) if 0 < pos.sum() < len(pos) else np.nan
        s[f"auc{thr}"] = auc
        s[f"cut{thr}"] = best[1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", type=int, default=120)
    ap.add_argument("--hybrid", type=int, default=60)
    a = ap.parse_args()
    files = sorted(glob.glob("/home/user/data/c18/*.npz"))[:: max(1, 300 // a.nights)][: a.nights]
    jobs = [(f, i < a.hybrid) for i, f in enumerate(files)]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(one, jobs))
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e04_apnea.json")
    summ = {k: summarize(res, k) for k in [f"psg_{d}" for d in DROPS] + [f"hyb_{d}" for d in DROPS]}
    json.dump(dict(nights=res, summary=summ), open(out, "w"), indent=1)
    for k, s in summ.items():
        if s:
            print(f"{k:9s} n={s['n']:3d} r={s['pearson']:.2f} ρ={s['spearman']:.2f} "
                  f"AUC15={s['auc15']:.2f} AUC30={s['auc30']:.2f} cut15={s['cut15']} cut30={s['cut30']}")


if __name__ == "__main__":
    main()
