"""Expérience 5 — Stades de sommeil : précision attendue et modèle final.

Validation croisée par nuit (5 plis, aucune nuit à la fois en apprentissage et en
test) sur 300 nuits PSG du PhysioNet/CinC Challenge 2018 :
  - « iPhone » : caractéristiques calculées par la chaîne iPhone complète sur des
    signaux d'accéléromètre/gyroscope simulés à partir de chaque nuit (X_hyb) ;
  - « PSG idéale » : mêmes caractéristiques depuis ceintures + ECG (X_psg).
Modèles : arbres boostés (contexte par fenêtres glissantes), GRU bidirectionnel
(contexte de toute la nuit) et leur moyenne (ensemble, retenu pour l'application).
Mesures : exactitude, κ de Cohen (2, 3, 4 classes), erreurs nuit par nuit (temps de
sommeil, efficacité, éveil intra-sommeil, minutes de chaque stade), effet des apnées.

    python experiments/e05_staging.py            (≈ 1 h sur 4 cœurs)
"""
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import model as M  # noqa: E402
from somno import seqmodel as S  # noqa: E402

FEATS = "/home/user/data/feats_c18.npz"
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
WEB_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "web", "model", "staging.json")
EXCLUDE = {"amp"}  # dépend de l'unité du capteur : seule amp_log (relative à la nuit) est gardée
GRU_EPOCHS = 25
W_GRU = 0.5


def feature_sets(names):
    allf = [n for n in names if n not in EXCLUDE]
    resp = [n for n in allf if not (n.startswith("hr") or n.startswith("mov") or n.startswith("t_"))]
    mov = [n for n in allf if n.startswith("mov") or n in ("t_since_mov", "t_to_mov")]
    hr = [n for n in allf if n.startswith("hr")]
    return {
        "respiration seule": resp,
        "respiration + mouvement": resp + mov,
        "respiration + cœur": resp + hr,
        "resp + cœur + mouvement": resp + hr + mov,
        "tout (+ heure de la nuit)": allf,
    }


def folds(night, k=5, seed=0):
    ids = np.unique(night)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return [np.isin(night, ids[i::k]) for i in range(k)]


def scores(y4, pred, night):
    ok = y4 >= 0
    k4, C4 = M.cohen_kappa(y4, pred, 4)
    y3 = np.where(y4 == 3, 2, np.where(y4 == 2, 1, y4))
    p3 = np.where(pred == 3, 2, np.where(pred == 2, 1, pred))
    k3, _ = M.cohen_kappa(y3, p3, 3)
    y2, p2 = (y4 > 0).astype(int), (pred > 0).astype(int)
    y2[~ok] = -1
    k2, C2 = M.cohen_kappa(y2, p2, 2)
    kn = [M.cohen_kappa(y4[night == n], pred[night == n], 4)[0] for n in np.unique(night)]
    return dict(acc4=float((pred[ok] == y4[ok]).mean()), kappa4=float(k4),
                kappa4_night_median=float(np.nanmedian(kn)),
                acc3=float((p3[ok] == y3[ok]).mean()), kappa3=float(k3),
                acc2=float((p2[ok] == y2[ok]).mean()), kappa2=float(k2),
                wake_sens=float(C2[0, 0] / C2[0].sum()), sleep_sens=float(C2[1, 1] / C2[1].sum()),
                recall4=[float(C4[i, i] / max(1, C4[i].sum())) for i in range(4)],
                confusion4=C4.astype(int).tolist())


def night_metrics(y4, p4, night):
    """Erreurs nuit par nuit : TST, efficacité, WASO, minutes de chaque stade."""
    rows = []
    for nid in np.unique(night):
        m = (night == nid) & (y4 >= 0)
        r = {}
        for nm, arr in (("psg", y4[m]), ("pred", p4[m])):
            sl = arr > 0
            r[nm + "_tst"] = sl.sum() / 2
            r[nm + "_se"] = sl.mean() * 100
            idx = np.flatnonzero(sl)
            r[nm + "_waso"] = ((~sl[idx[0]:idx[-1] + 1]).sum() / 2) if len(idx) else 0
            for s, lab in enumerate(["wake", "light", "deep", "rem"]):
                r[f"{nm}_{lab}"] = (arr == s).sum() / 2
        rows.append(r)
    out = {}
    for key in ["tst", "se", "waso", "light", "deep", "rem"]:
        a = np.array([r["psg_" + key] for r in rows])
        b = np.array([r["pred_" + key] for r in rows])
        d = b - a
        out[key] = dict(bias=float(d.mean()), sd=float(d.std()), mae=float(np.abs(d).mean()),
                        r=float(np.corrcoef(a, b)[0, 1]), psg=a.tolist(), pred=b.tolist())
    return out


def seqs_for(X, names, night, ids, mu, sd, y4):
    xs, ys = [], []
    for n in ids:
        m = night == n
        xs.append(S.make_inputs(X[m], names, mu, sd)[0])
        ys.append(y4[m])
    return xs, ys


def cv_models(X, y4, night, names, cols, label, with_gru=True):
    idx = [names.index(c) for c in cols]
    Ph = np.zeros((len(y4), 4))
    Pg = np.zeros((len(y4), 4))
    t0 = time.time()
    for k, test in enumerate(folds(night)):
        tr = (~test) & (y4 >= 0)
        clf = M.train_hgb(X[tr][:, idx], y4[tr], max_iter=200, class_weight=None)
        Ph[test] = clf.predict_proba(X[test][:, idx])
        if with_gru:
            _, mu, sd = S.make_inputs(X[~test], names)
            tr_ids = np.unique(night[~test])
            xs, ys = seqs_for(X, names, night, tr_ids, mu, sd, y4)
            net = S.train(xs, ys, epochs=GRU_EPOCHS, seed=k)
            for n in np.unique(night[test]):
                m = night == n
                Pg[m] = S.predict(net, S.make_inputs(X[m], names, mu, sd)[0])
        print(f"  [{label}] pli {k + 1}/5 ({time.time() - t0:.0f}s)", flush=True)
    res = {"HGB": (Ph, scores(y4, Ph.argmax(1), night))}
    if with_gru:
        Pe = (1 - W_GRU) * Ph + W_GRU * Pg
        res["GRU"] = (Pg, scores(y4, Pg.argmax(1), night))
        res["ensemble"] = (Pe, scores(y4, Pe.argmax(1), night))
    for k, (_, s) in res.items():
        print(f"[{label}] {k:9s} acc4={s['acc4']:.3f} κ4={s['kappa4']:.3f} (méd. nuit {s['kappa4_night_median']:.2f}) "
              f"κ3={s['kappa3']:.3f} acc2={s['acc2']:.3f} κ2={s['kappa2']:.3f} "
              f"éveil={s['wake_sens']:.2f} sommeil={s['sleep_sens']:.2f}", flush=True)
    return res


def main():
    d = np.load(FEATS, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    y4 = M.to4(d["y"])
    night = d["night"]
    Xh, Xp = d["Xh"].astype(np.float64), d["Xp"].astype(np.float64)
    ahi = d["ahi"]
    print(f"{len(np.unique(night))} nuits, {len(y4)} époques, répartition",
          np.round(np.bincount(y4[y4 >= 0]) / (y4 >= 0).sum(), 3), flush=True)
    fsets = feature_sets(names)
    cols = fsets["tout (+ heure de la nuit)"]
    results = {"ablation_hgb_iphone": {}}
    for name, c in fsets.items():
        r = cv_models(Xh, y4, night, names, c, f"iPhone · {name}", with_gru=False)
        results["ablation_hgb_iphone"][name] = r["HGB"][1]
    rh = cv_models(Xh, y4, night, names, cols, "iPhone")
    results["iphone"] = {k: v[1] for k, v in rh.items()}
    Pe = rh["ensemble"][0]
    pred = Pe.argmax(1)
    results["iphone"]["ensemble"]["per_night"] = night_metrics(y4, pred, night)
    rp = cv_models(Xp, y4, night, names, cols, "PSG idéale")
    results["psg"] = {k: v[1] for k, v in rp.items()}
    # effet des apnées et du couplage matelas
    per = []
    for i, n in enumerate(np.unique(night)):
        m = night == n
        per.append(dict(night=int(n), ahi=float(ahi[i]), tilt=float(d["cond"][i][0]), bcg=float(d["cond"][i][1]),
                        kappa=float(M.cohen_kappa(y4[m], pred[m], 4)[0]),
                        acc=float((pred[m] == y4[m])[y4[m] >= 0].mean())))
    results["per_night"] = per
    groups = {}
    for lab, lo, hi in (("IAH < 5", 0, 5), ("IAH 5–15", 5, 15), ("IAH 15–30", 15, 30), ("IAH ≥ 30", 30, 1e9)):
        ids = [p["night"] for p in per if lo <= p["ahi"] < hi]
        if ids:
            m = np.isin(night, ids)
            groups[lab] = dict(n=len(ids), **{k: v for k, v in scores(y4[m], pred[m], night[m]).items()
                                              if k in ("acc4", "kappa4", "acc2", "kappa2")})
    results["by_ahi"] = groups
    print("par IAH :", json.dumps(groups, ensure_ascii=False), flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "e05_pred_iphone.npy"), Pe.astype(np.float32))
    json.dump(results, open(os.path.join(OUT, "e05_staging.json"), "w"), indent=1, ensure_ascii=False)

    # ---- modèle final : toutes les nuits, caractéristiques iPhone + PSG (augmentation)
    print("modèle final…", flush=True)
    idx = [names.index(c) for c in cols]
    ok = y4 >= 0
    clf = M.train_hgb(np.concatenate([Xh[ok][:, idx], Xp[ok][:, idx]]), np.concatenate([y4[ok], y4[ok]]),
                      max_iter=200, class_weight=None)
    hgb = M.export_hgb(clf, cols)
    _, mu, sd = S.make_inputs(np.concatenate([Xh, Xp]), names)
    ids = np.unique(night)
    xs1, ys1 = seqs_for(Xh, names, night, ids, mu, sd, y4)
    xs2, ys2 = seqs_for(Xp, names, night, ids, mu, sd, y4)
    net = S.train(xs1 + xs2, ys1 + ys2, epochs=GRU_EPOCHS, seed=42)
    gru = S.export(net, mu, sd)
    s = results["iphone"]["ensemble"]
    mdl = dict(kind="ensemble", stages=M.STAGES4, w_gru=W_GRU, hgb=hgb, gru=gru,
               trained_on=f"{len(ids)} nuits PSG (PhysioNet/CinC Challenge 2018), signaux iPhone simulés + PSG",
               cv_iphone={k: s[k] for k in ("acc4", "kappa4", "acc2", "kappa2")})
    os.makedirs(os.path.dirname(WEB_MODEL), exist_ok=True)
    M.save(mdl, WEB_MODEL)
    print("modèle exporté :", WEB_MODEL, os.path.getsize(WEB_MODEL) // 1024, "Ko", flush=True)


if __name__ == "__main__":
    main()
