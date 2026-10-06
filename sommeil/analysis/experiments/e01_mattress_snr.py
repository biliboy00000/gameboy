"""Expérience 1 — Précision respiration / fréquence cardiaque d'un iPhone sur le matelas
selon la force du signal et la résolution des capteurs.

Pour chaque nuit PSG réelle, on synthétise ce que l'iPhone aurait mesuré (respiration
réelle des ceintures, battements réels de l'ECG) avec :
  - inclinaison respiratoire du matelas : 0,01° (téléphone loin / matelas ferme) → 0,3°
  - amplitude BCG : 0,1 mg (loin du thorax) → 3 mg (sous l'oreiller/le thorax)
  - quantification : aucune (app native), 1 mg (Safari ?), 0,1 m/s² (arrondi « anti-
    empreinte » de Chrome)
puis on compare, sur les époques sans mouvement, la fréquence respiratoire et la FC
estimées par la chaîne iPhone à la vérité PSG.

    python experiments/e01_mattress_snr.py --nights 4
"""
import argparse
import glob
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import heart, pipeline  # noqa: E402

TILTS = [0.01, 0.03, 0.1, 0.3]
BCGS = [0.1, 0.3, 1.0, 3.0]
QUANT = {"natif": (0.0, 0.0), "1 mg": (0.0098, 0.0005), "0,1 m/s²": (0.1, 0.1 * np.pi / 180)}


def truth(night, rp):
    Xp, names = pipeline.psg_features(night, rp)
    return Xp[:, names.index("rr")], Xp[:, names.index("hr")], Xp[:, names.index("mov_frac")]


def run(args):
    f, tilt, bcg, qname, seed = args
    night = dict(np.load(f))
    rp = heart.detect_r_peaks(night["ecg"], 200.0) / 200.0
    rr_t, hr_t, mv = truth(night, rp)
    qa, qg = QUANT[qname]
    X, names, _ = pipeline.hybrid_features(night, rp, seed=seed, resp_tilt_deg=tilt, bcg_mg=bcg,
                                           quant_acc=qa, quant_gyro=qg)
    n = min(len(X), len(rr_t))
    still = (X[:n, names.index("mov_frac")] == 0) & (mv[:n] == 0) & (night["stages"][:n] > 0)
    rr = X[:n, names.index("rr")]
    hr = X[:n, names.index("hr")]
    ok_r = still & np.isfinite(rr_t[:n])
    ok_h = still & np.isfinite(hr_t[:n])
    er = np.abs(rr - rr_t[:n])[ok_r]
    eh = np.abs(hr - hr_t[:n])[ok_h]
    if len(er) == 0 or len(eh) == 0:
        return dict(file=os.path.basename(f), tilt=tilt, bcg=bcg, quant=qname, rr_mae=np.nan,
                    rr_cov=np.nan, rr_within1=np.nan, hr_mae=np.nan, hr_cov=np.nan, hr_within3=np.nan)
    return dict(file=os.path.basename(f), tilt=tilt, bcg=bcg, quant=qname,
                rr_mae=float(np.nanmean(er)), rr_cov=float(np.isfinite(er).mean()),
                rr_within1=float(np.nanmean(er <= 1.0)),
                hr_mae=float(np.nanmean(eh)), hr_cov=float(np.isfinite(eh).mean()),
                hr_within3=float(np.mean(np.where(np.isfinite(eh), eh <= 3.0, False))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nights", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "e01_snr.json"))
    a = ap.parse_args()
    files = sorted(glob.glob("/home/user/data/c18/*.npz"))[:: 300 // a.nights][: a.nights]
    jobs = [(f, t, b, q, i) for i, (f, (t, b), q) in
            enumerate(itertools.product(files, zip(TILTS, BCGS), QUANT))]
    # grille complète tilt × bcg seulement en natif, pour la carte
    jobs += [(f, t, b, "natif", 100 + i) for i, (f, t, b) in
             enumerate(itertools.product(files, TILTS, BCGS)) if TILTS.index(t) != BCGS.index(b)]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(run, jobs))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["tilt"], r["bcg"], r["quant"])].append(r)
    print(f"{'inclin.':>8} {'BCG':>5} {'quant':>9} | {'FR err':>7} {'FR ±1':>6} | {'FC err':>7} {'FC ±3':>6} {'couv':>5}")
    for k in sorted(agg):
        rs = agg[k]
        m = lambda key: np.nanmean([r[key] for r in rs])
        print(f"{k[0]:8.2f} {k[1]:5.1f} {k[2]:>9} | {m('rr_mae'):7.2f} {m('rr_within1'):6.1%} | "
              f"{m('hr_mae'):7.2f} {m('hr_within3'):6.1%} {m('hr_cov'):5.0%}")


if __name__ == "__main__":
    main()
