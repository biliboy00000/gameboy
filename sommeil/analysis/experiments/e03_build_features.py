"""Expérience 3a — Construit la base d'apprentissage des stades de sommeil.

Pour chaque nuit PSG (PhysioNet Challenge 2018) :
  X_psg : caractéristiques « idéales » (ceintures + ECG) — borne haute ;
  X_hyb : caractéristiques obtenues par la chaîne iPhone complète appliquée à des
          signaux d'accéléromètre/gyroscope simulés depuis cette nuit, avec un
          couplage matelas tiré au hasard (inclinaison 0,03–0,5°, BCG 0,2–3 mg,
          quantification 1 mg une nuit sur deux, partenaire qui bouge une nuit sur
          cinq) → reflète la diversité des lits et des placements.

    python experiments/e03_build_features.py --out /home/user/data/feats_c18.npz
"""
import argparse
import glob
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import heart, pipeline  # noqa: E402


def apnea_epochs(night, n_ep):
    """Époques recouvertes par un événement respiratoire (apnée/hypopnée) PSG."""
    flag = np.zeros(n_ep, dtype=np.int8)
    for s, e, t in zip(night["ev_start"], night["ev_end"], night["ev_type"]):
        a, b = int(s // 30), int(e // 30)
        flag[a:min(n_ep, b + 1)] = 1
    return flag


def one(path):
    rid = os.path.basename(path)[:-4]
    seed = zlib.crc32(rid.encode())
    rng = np.random.default_rng(seed)
    night = dict(np.load(path))
    fs_ecg = float(night["fs_ecg"])
    rp = heart.detect_r_peaks(night["ecg"], fs_ecg) / fs_ecg
    Xp, names = pipeline.psg_features(night, rp)
    kw = dict(resp_tilt_deg=float(np.exp(rng.uniform(np.log(0.03), np.log(0.5)))),
              bcg_mg=float(np.exp(rng.uniform(np.log(0.2), np.log(3.0)))),
              quant_acc=0.0098 if rng.random() < 0.5 else 0.0,
              partner_rate_h=6.0 if rng.random() < 0.2 else 0.0)
    Xh, names_h, info = pipeline.hybrid_features(night, rp, seed=seed % 10000, **kw)
    assert names == names_h
    n = min(len(Xp), len(Xh), len(night["stages"]))
    tst_h = (night["stages"] > 0).sum() * 30 / 3600
    events = night["ev_type"]
    n_ah = int(np.isin(events, ["obstructiveapnea", "centralapnea", "mixedapnea", "hypopnea"]).sum())
    return dict(rid=rid, Xp=Xp[:n].astype(np.float32), Xh=Xh[:n].astype(np.float32),
                y=night["stages"][:n], apnea=apnea_epochs(night, n), names=names,
                ahi=n_ah / max(tst_h, 0.5), cond=[kw["resp_tilt_deg"], kw["bcg_mg"],
                                                  kw["quant_acc"], kw["partner_rate_h"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/user/data/c18")
    ap.add_argument("--out", default="/home/user/data/feats_c18.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "*.npz")))
    if a.limit:
        files = files[: a.limit]
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, files)):
            res.append(r)
            if i % 10 == 0:
                print(f"{i + 1}/{len(files)} {r['rid']} ahi={r['ahi']:.1f}", flush=True)
    night_id = np.concatenate([np.full(len(r["y"]), i) for i, r in enumerate(res)])
    np.savez_compressed(
        a.out, Xp=np.concatenate([r["Xp"] for r in res]), Xh=np.concatenate([r["Xh"] for r in res]),
        y=np.concatenate([r["y"] for r in res]), apnea=np.concatenate([r["apnea"] for r in res]),
        night=night_id, names=np.array(res[0]["names"]), rids=np.array([r["rid"] for r in res]),
        ahi=np.array([r["ahi"] for r in res]), cond=np.array([r["cond"] for r in res]))
    print("ok", a.out, len(night_id), "époques")


if __name__ == "__main__":
    main()
