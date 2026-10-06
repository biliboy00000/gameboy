"""Figures des expériences → sommeil/docs/figures/*.png

    python experiments/make_figures.py
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import model as M  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "..", "docs", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STAGE = {0: "#f2b84b", 1: "#5fa8ff", 2: "#2f5bd8", 3: "#b48cff"}
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "axes.titlecolor": INK, "axes.titlesize": 11, "lines.linewidth": 2,
})


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_snr():
    r = load("e01_snr.json")
    if not r:
        return
    tilts = sorted({x["tilt"] for x in r})
    bcgs = sorted({x["bcg"] for x in r})
    nat = [x for x in r if x["quant"] == "natif"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (key, title, vmax) in enumerate((("rr_mae", "Erreur fréquence respiratoire (resp/min)", 6),
                                             ("hr_mae", "Erreur fréquence cardiaque (bpm)", 15))):
        Z = np.full((len(tilts), len(bcgs)), np.nan)
        for i, t in enumerate(tilts):
            for j, b in enumerate(bcgs):
                v = [x[key] for x in nat if x["tilt"] == t and x["bcg"] == b and x[key] == x[key]]
                if v:
                    Z[i, j] = np.mean(v)
        ax[k].imshow(Z, cmap="Blues_r", vmin=0, vmax=vmax, aspect="auto", origin="lower")
        ax[k].grid(False)
        for i in range(len(tilts)):
            for j in range(len(bcgs)):
                if np.isfinite(Z[i, j]):
                    ax[k].text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center",
                               color="white" if Z[i, j] < vmax * 0.45 else INK, fontsize=9)
        ax[k].set_xticks(range(len(bcgs)), [f"{b:g} mg" for b in bcgs])
        ax[k].set_yticks(range(len(tilts)), [f"{t:g}°" for t in tilts])
        ax[k].set_xlabel("amplitude du ballistocardiogramme au téléphone")
        ax[k].set_ylabel("inclinaison respiratoire du matelas")
        ax[k].set_title(title)
    fig.suptitle("iPhone sur le matelas — précision selon le couplage (6 nuits PSG réelles, époques calmes)",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e01_matelas_snr.png"), dpi=130)
    plt.close(fig)


def fig_sonar():
    r = load("e02_sonar.json")
    if not r:
        return
    dists = sorted({x["dist"] for x in r})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for nz, col in zip(sorted({x["noise"] for x in r}), (BLUE, AQUA, ORANGE)):
        for bl, ls in ((0.0, "-"), (10.0, "--")):
            sel = [[x for x in r if x["dist"] == d and x["noise"] == nz and x["blanket"] == bl] for d in dists]
            ax[0].plot(dists, [np.nanmean([x["rr_within1"] for x in s]) * 100 for s in sel], ls, color=col,
                       marker="o", ms=5, label=f"bruit {nz} dB" + (" + couette" if bl else ""))
            ax[1].plot(dists, [np.nanmean([x["hr_within3"] for x in s]) * 100 for s in sel], ls, color=col,
                       marker="o", ms=5)
    ax[0].set_title("Respiration : époques à ±1 resp/min (%)")
    ax[1].set_title("Pouls : époques à ±3 bpm (%)")
    for a in ax:
        a.set_xlabel("distance iPhone → thorax (m)")
        a.set_ylim(0, 102)
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Sonar FMCW 18–20,5 kHz — respiration et battements réels de 5 nuits PSG", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e02_sonar.png"), dpi=130)
    plt.close(fig)


def fig_staging():
    r = load("e05_staging.json")
    if not r:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    rows = [(f"Arbres · {k}", v["kappa4"], MUTED) for k, v in r["ablation_hgb_iphone"].items()]
    rows += [("iPhone · GRU bidirectionnel", r["iphone"]["GRU"]["kappa4"], BLUE),
             ("iPhone · ensemble (retenu)", r["iphone"]["ensemble"]["kappa4"], BLUE),
             ("PSG idéale · ensemble", r["psg"]["ensemble"]["kappa4"], ORANGE)]
    y = np.arange(len(rows))[::-1]
    ax[0].barh(y, [x[1] for x in rows], color=[x[2] for x in rows], height=0.62)
    for yy, (lab, v, _) in zip(y, rows):
        ax[0].text(v + 0.005, yy, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax[0].set_yticks(y, [x[0] for x in rows], fontsize=9)
    ax[0].set_xlabel("κ de Cohen, 4 stades (0 = hasard, 1 = parfait)")
    ax[0].set_xlim(0, max(x[1] for x in rows) + 0.08)
    ax[0].set_title("Apport de chaque signal (validation croisée, 300 nuits)")
    C = np.array(r["iphone"]["ensemble"]["confusion4"], dtype=float)
    Cn = C / C.sum(axis=1, keepdims=True)
    ax[1].imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax[1].grid(False)
    for i in range(4):
        for j in range(4):
            ax[1].text(j, i, f"{Cn[i, j] * 100:.0f} %", ha="center", va="center",
                       color="white" if Cn[i, j] > 0.5 else INK, fontsize=9)
    ax[1].set_xticks(range(4), M.STAGES4)
    ax[1].set_yticks(range(4), M.STAGES4)
    ax[1].set_xlabel("stade estimé (iPhone)")
    ax[1].set_ylabel("stade PSG (vérité)")
    ax[1].set_title("Confusion (lignes = 100 %)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e05_stades.png"), dpi=130)
    plt.close(fig)

    pn = r["iphone"]["ensemble"]["per_night"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for a, key, lab in zip(ax, ("tst", "deep", "rem"), ("temps de sommeil", "sommeil profond", "REM")):
        a.scatter(pn[key]["psg"], pn[key]["pred"], s=10, color=BLUE, alpha=0.6, edgecolors="none")
        lim = max(max(pn[key]["psg"]), max(pn[key]["pred"])) * 1.05
        a.plot([0, lim], [0, lim], color=MUTED, lw=1)
        a.set_xlim(0, lim)
        a.set_ylim(0, lim)
        a.set_xlabel("PSG (min)")
        a.set_ylabel("iPhone (min)")
        a.set_title(f"{lab} : r = {pn[key]['r']:.2f}, écart {pn[key]['bias']:+.0f} ± {pn[key]['sd']:.0f} min", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e05_nuits.png"), dpi=130)
    plt.close(fig)


def fig_example():
    p = os.path.join(RES, "e05_pred_iphone.npy")
    r = load("e05_staging.json")
    if not (os.path.exists(p) and r):
        return
    P = np.load(p)
    d = np.load("/home/user/data/feats_c18.npz")
    y4 = M.to4(d["y"])
    night = d["night"]
    per = sorted(r["per_night"], key=lambda x: x["kappa"])
    ex = per[len(per) // 2]["night"]
    m = night == ex
    yt, yp = y4[m], P[m].argmax(1)
    row = {0: 3, 3: 2, 1: 1, 2: 0}
    fig, ax = plt.subplots(2, 1, figsize=(11, 3.8), sharex=True)
    for a, yy, title in ((ax[0], yt, "PSG (scorée par des experts)"), (ax[1], yp, "iPhone sur le matelas (estimation)")):
        t = np.arange(len(yy)) / 120
        for s in range(4):
            sel = yy == s
            a.bar(t[sel], 0.8, bottom=np.full(sel.sum(), row[s] - 0.4), width=1 / 120, color=STAGE[s])
        a.set_yticks([3, 2, 1, 0], ["Éveil", "REM", "Léger", "Profond"])
        a.set_title(title, loc="left", fontsize=10)
        a.grid(False)
    ax[1].set_xlabel("heures depuis le coucher")
    k = per[len(per) // 2]["kappa"]
    fig.suptitle(f"Nuit médiane de la validation (κ = {k:.2f})", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e05_exemple_nuit.png"), dpi=130)
    plt.close(fig)


def fig_apnea():
    r = load("e04_apnea.json")
    if not r:
        return
    s = r["summary"]
    best = max((k for k in s if k.startswith("hyb") and s[k]), key=lambda k: s[k]["auc15"], default=None)
    if not best:
        return
    ns = [x for x in r["nights"] if best in x]
    ahi = np.array([x["ahi"] for x in ns])
    est = np.array([x[best] for x in ns])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(ahi, est, s=14, color=BLUE, alpha=0.7, edgecolors="none")
    ax.axvline(15, color=ORANGE, lw=1, ls="--")
    ax.set_xlabel("IAH de la polysomnographie (événements/h)")
    ax.set_ylabel("indice estimé par l'iPhone (/h)")
    ss = s[best]
    ax.set_title(f"Apnées : ρ = {ss['spearman']:.2f}, AUC (IAH ≥ 15) = {ss['auc15']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e04_apnees.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_snr, fig_sonar, fig_staging, fig_example, fig_apnea):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:  # une figure manquante ne bloque pas les autres
            print("échec", f.__name__, repr(e))
