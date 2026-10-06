"""Analyse hors ligne d'une nuit exportée (.somno) avec le modèle de l'application.

    python -m somno.cli nuit.somno --out rapport.png [--csv epoques.csv]

Utile pour ré-analyser une nuit avec un algorithme amélioré, comparer à un autre
appareil, ou explorer les signaux bruts (respiration, BCG, mouvements).
"""
import argparse
import datetime as dt
import json
import os

import numpy as np

from . import apnea, io, model as M, pipeline, seqmodel as S

HERE = os.path.dirname(__file__)
DEFAULT_MODEL = os.path.join(HERE, "..", "..", "web", "model", "staging.json")
COLORS = {0: "#f2b84b", 3: "#b48cff", 1: "#5fa8ff", 2: "#2f5bd8"}


def stage_proba(mdl, X, names):
    idx = [names.index(c) for c in mdl["hgb"]["features"]]
    Ph = M.predict_json(mdl["hgb"], X[:, idx])
    if "gru" not in mdl:
        return Ph
    Pg = S.predict_json(mdl["gru"], X, names)
    w = mdl.get("w_gru", 0.5)
    return (1 - w) * Ph + w * Pg


def analyze(path, model_path=DEFAULT_MODEL):
    d = io.read_somno(path)
    mo = d["motion"]
    fs = float(mo["rate"])
    acc, gyro = mo["data"][:, :3], mo["data"][:, 3:6]
    X, names, sig = pipeline.phone_features(acc, gyro, fs, return_signals=True)
    mdl = json.load(open(model_path))
    P = stage_proba(mdl, X, names)
    hyp = P.argmax(1)
    ev = apnea.detect_events(sig["resp_wave"], sig["moving"])
    idx_rdi, n_ev = apnea.rdi(ev, hyp > 0)
    ahi_est = None
    ap_path = os.path.join(os.path.dirname(model_path), "apnea.json")
    if os.path.exists(ap_path):
        am = json.load(open(ap_path))
        pa = M.predict_json(am, X[:, [names.index(c) for c in am["features"]]])[:, 1]
        sl = hyp > 0
        if sl.any():
            ahi_est = max(0.0, am["calib"][0] * pa[sl].mean() * 120 + am["calib"][1])
    return dict(meta=d["meta"], X=X, names=names, P=P, hyp=hyp, sig=sig, events=ev, rdi=idx_rdi, n_events=n_ev,
                ahi_est=ahi_est)


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hyp, X, names = res["hyp"], res["X"], res["names"]
    t0 = dt.datetime.fromtimestamp(res["meta"]["start"] / 1000)
    t = [t0 + dt.timedelta(seconds=30 * i) for i in range(len(hyp))]
    row = {0: 3, 3: 2, 1: 1, 2: 0}
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1, 1]))
    for i, s in enumerate(hyp):
        ax[0].barh(row[s], 30 / 86400, left=matplotlib.dates.date2num(t[i]), color=COLORS[s], height=0.7)
    ax[0].set_yticks([3, 2, 1, 0], ["Éveil", "REM", "Léger", "Profond"])
    iah = f", IAH estimé {res['ahi_est']:.0f}/h" if res["ahi_est"] is not None else ""
    ax[0].set_title(f"Nuit du {t0:%d/%m/%Y} — sommeil {(hyp > 0).sum() / 120:.1f} h{iah}")
    ax[1].plot(t, X[:, names.index("rr")], ".", ms=2, color="#5fa8ff")
    ax[1].set_ylabel("resp./min")
    ax[2].plot(t, X[:, names.index("hr")], ".", ms=2, color="#f26d6d")
    ax[2].set_ylabel("bpm")
    ax[3].bar(t, X[:, names.index("mov_frac")], width=30 / 86400, color="#f2b84b")
    ax[3].set_ylabel("mouvement")
    ax[3].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("somno")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="figure PNG")
    ap.add_argument("--csv", default=None, help="caractéristiques par époque")
    a = ap.parse_args()
    res = analyze(a.somno, a.model)
    hyp = res["hyp"]
    print(f"{len(hyp)} époques · sommeil {(hyp > 0).sum() / 2:.0f} min · "
          + " · ".join(f"{n} {(hyp == k).sum() / 2:.0f} min" for k, n in enumerate(M.STAGES4))
          + (f" · IAH estimé {res['ahi_est']:.0f}/h" if res["ahi_est"] is not None else "")
          + f" · chutes d'amplitude {res['rdi']:.1f}/h")
    if a.out:
        plot(res, a.out)
        print("figure :", a.out)
    if a.csv:
        cols = ["stade"] + [f"p_{s}" for s in M.STAGES4] + res["names"]
        rows = np.column_stack([hyp, res["P"], res["X"]])
        np.savetxt(a.csv, rows, delimiter=";", header=";".join(cols), comments="", fmt="%.5g")
        print("CSV :", a.csv)


if __name__ == "__main__":
    main()
