"""Expérience 7 — Test final sur des nuits jamais vues.

Le modèle exporté pour l'application (web/model/staging.json) a été entraîné sur 300 nuits.
On télécharge d'autres nuits du Challenge 2018, absentes de cet ensemble, on les convertit
en signaux d'iPhone sur le matelas et on applique la chaîne complète avec le modèle final,
exactement comme le fera l'application. Une nuit est aussi exportée en .f32 pour le test
de bout en bout dans le navigateur (capture d'écran du rapport).

    python experiments/e07_holdout.py --n 12
"""
import argparse
import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "datasets"))
from somno import cli, heart, model as M, pipeline  # noqa: E402
import fetch_challenge2018 as F  # noqa: E402

TRAIN_DIR = "/home/user/data/c18"
OUT_DIR = "/home/user/data/c18_holdout"


def one(rec):
    F.convert(rec, OUT_DIR)
    night = dict(np.load(os.path.join(OUT_DIR, rec + ".npz")))
    rp = heart.detect_r_peaks(night["ecg"], 200.0) / 200.0
    seed = zlib.crc32(rec.encode())
    rng = np.random.default_rng(seed)
    kw = dict(resp_tilt_deg=float(np.exp(rng.uniform(np.log(0.05), np.log(0.4)))),
              bcg_mg=float(np.exp(rng.uniform(np.log(0.3), np.log(2.0)))))
    acc, gyro, _ = pipeline.hybrid_signals(night, rp, seed=seed % 10000, **kw)
    X, names = pipeline.phone_features(acc, gyro, 50.0)
    mdl = json.load(open(cli.DEFAULT_MODEL))
    P = cli.stage_proba(mdl, X, names)
    n = min(len(P), len(night["stages"]))
    y4 = M.to4(night["stages"][:n])
    return dict(rec=rec, y=y4.tolist(), pred=P[:n].argmax(1).tolist(), cond=kw,
                signals=(acc.astype(np.float32), gyro.astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = [r.strip().strip("/") for r in open(os.path.join(TRAIN_DIR, "RECORDS")) if r.strip()]
    used = {f[:-4] for f in os.listdir(TRAIN_DIR) if f.endswith(".npz")}
    cand = [r for r in recs[1::7] if r not in used][: a.n]
    with ProcessPoolExecutor(3) as ex:
        res = list(ex.map(one, cand))
    y = np.concatenate([r["y"] for r in res])
    p = np.concatenate([r["pred"] for r in res])
    ok = y >= 0
    k4, C = M.cohen_kappa(y, p, 4)
    y2 = np.where(ok, (y > 0).astype(int), -1)
    k2, _ = M.cohen_kappa(y2, (p > 0).astype(int), 2)
    per = []
    for r in res:
        yy, pp = np.array(r["y"]), np.array(r["pred"])
        per.append(dict(rec=r["rec"], kappa=float(M.cohen_kappa(yy, pp, 4)[0]),
                        acc=float((yy == pp)[yy >= 0].mean()), tst_psg=float((yy > 0).sum() / 2),
                        tst_pred=float((pp > 0).sum() / 2), cond=r["cond"]))
    summ = dict(n=len(res), acc4=float((p[ok] == y[ok]).mean()), kappa4=float(k4),
                acc2=float(((p > 0) == (y > 0))[ok].mean()), kappa2=float(k2), confusion=C.astype(int).tolist(),
                per_night=per)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_night"}, indent=1))
    for q in per:
        print(f"  {q['rec']}: κ={q['kappa']:.2f} acc={q['acc']:.2f} TST PSG {q['tst_psg']:.0f} min / iPhone {q['tst_pred']:.0f} min")
    json.dump(summ, open(os.path.join(HERE, "..", "results", "e07_holdout.json"), "w"), indent=1)
    # nuit médiane → fichiers pour le test navigateur
    med = sorted(range(len(per)), key=lambda i: per[i]["kappa"])[len(per) // 2]
    acc, gyro = res[med]["signals"]
    np.concatenate([acc, gyro], axis=1).astype(np.float32).tofile("/home/user/data/holdout_night.f32")
    stages = (np.array(res[med]["y"]) + 0).tolist()
    # au format AASM attendu par le test e2e (0 W, 1 N1, 2 N2, 3 N3, 4 R) via la table inverse
    inv = {0: 0, 1: 2, 2: 3, 3: 4, -1: 0}
    json.dump(dict(stages=[inv[s] for s in stages], rec=res[med]["rec"]), open("/home/user/data/holdout_night.json", "w"))
    print("nuit exportée pour le navigateur :", res[med]["rec"])


if __name__ == "__main__":
    main()
