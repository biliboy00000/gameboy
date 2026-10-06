"""Expérience 4b — Dépistage d'apnées appris (remplace le simple seuil d'amplitude).

Un classifieur (arbres boostés) estime, pour chaque époque de sommeil, la probabilité
qu'elle contienne une apnée ou une hypopnée, à partir des mêmes caractéristiques que
les stades : variabilité d'amplitude et de rythme respiratoire, cycles de la fréquence
cardiaque, micro-éveils (mouvements). La moyenne sur la nuit, calibrée linéairement sur
l'IAH (validation croisée), donne un indice estimé en événements par heure.

    python experiments/e04b_apnea_ml.py
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import model as M  # noqa: E402

HERE = os.path.dirname(__file__)
OUT_MODEL = os.path.join(HERE, "..", "..", "web", "model", "apnea.json")


def train(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=24,
                                          min_samples_leaf=80, l2_regularization=1.0, random_state=0).fit(X, y)


def night_index(P, night, sleep):
    ids = np.unique(night)
    return np.array([P[(night == n) & sleep].mean() * 120 if ((night == n) & sleep).any() else 0.0 for n in ids])


def main():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    d = np.load("/home/user/data/feats_c18.npz")
    names = [str(n) for n in d["names"]]
    cols = [n for n in names if n != "amp"]
    idx = [names.index(c) for c in cols]
    night, ap, y, ahi = d["night"], d["apnea"], d["y"], d["ahi"]
    sleep = y > 0
    res = {}
    for key, label in (("Xh", "iphone"), ("Xp", "psg")):
        X = d[key].astype(float)[:, idx]
        P = np.zeros(len(y))
        ids = np.unique(night)
        rng = np.random.default_rng(0)
        rng.shuffle(ids)
        for k in range(5):
            te = np.isin(night, ids[k::5])
            tr = (~te) & sleep
            P[te] = train(X[tr], ap[tr]).predict_proba(X[te])[:, 1]
        est = night_index(P, night, sleep)
        a = ahi[np.unique(night)]
        # calibration linéaire de l'indice sur l'IAH (moindres carrés sur la validation croisée)
        A = np.stack([est, np.ones_like(est)], 1)
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        cal = np.maximum(0, A @ coef)
        out = dict(epoch_auc=float(roc_auc_score(ap[sleep], P[sleep])), spearman=float(spearmanr(a, est).statistic),
                   pearson_cal=float(np.corrcoef(a, cal)[0, 1]), auc15=float(roc_auc_score(a >= 15, est)),
                   auc30=float(roc_auc_score(a >= 30, est)), calib=coef.tolist(),
                   mae_cal=float(np.mean(np.abs(cal - a))))
        for thr in (15, 30):
            pos = a >= thr
            out[f"sens{thr}"] = float(np.mean(cal[pos] >= thr))
            out[f"spec{thr}"] = float(np.mean(cal[~pos] < thr))
        out["ahi"] = a.tolist()
        out["est_cal"] = cal.tolist()
        res[label] = out
        print(label, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k not in ("ahi", "est_cal")},
              flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "results", "e04b_apnea_ml.json"), "w"), indent=1)
    # modèle final (iPhone + PSG), calibration issue de la validation croisée iPhone
    Xall = np.concatenate([d["Xh"].astype(float)[:, idx][sleep], d["Xp"].astype(float)[:, idx][sleep]])
    yall = np.concatenate([ap[sleep], ap[sleep]])
    clf = train(Xall, yall)
    mdl = M.export_hgb(clf, cols, extra=dict(calib=res["iphone"]["calib"],
                                             note="P(époque avec apnée/hypopnée) ; indice = moyenne(P)·120 puis calibration linéaire vers l'IAH",
                                             cv=dict((k, res["iphone"][k]) for k in ("spearman", "auc15", "auc30", "sens15", "spec15"))))
    M.save(mdl, OUT_MODEL)
    print("modèle apnées :", OUT_MODEL, os.path.getsize(OUT_MODEL) // 1024, "Ko")


if __name__ == "__main__":
    main()
