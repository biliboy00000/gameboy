"""Expérience 2 — Sonar FMCW (iPhone sur la table de nuit, sans contact).

Respiration et battements RÉELS de nuits PSG (segments de 20 min de sommeil) →
mouvement de la poitrine (2–8 mm crête-à-crête + 0,2–0,4 mm par battement) →
simulation des échos 18–20,5 kHz (fouillis de la chambre, couette qui atténue,
bruit du micro) → démodulation → fréquence respiratoire et FC, comparées à la PSG.

    python experiments/e02_sonar.py
"""
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
from somno import dsp, heart, respiration, sonar  # noqa: E402

DISTS = [0.5, 1.0, 1.5, 2.0]
NOISE = [-80, -70, -60]
BLANKET = [0.0, 10.0]
SEG_S = 1200


def segment(path, rng):
    night = dict(np.load(path))
    st = night["stages"]
    # premier bloc de 20 min de sommeil continu (sans éveil)
    ne = SEG_S // 30
    for e0 in range(0, len(st) - ne, 10):
        if (st[e0:e0 + ne] >= 1).all():
            break
    t0 = e0 * 30
    fs_r = float(night["fs_resp"])
    resp = (night["chest"] + night["abd"]).astype(np.float64)[int(t0 * fs_r): int((t0 + SEG_S) * fs_r)]
    resp = dsp.filtfilt(resp, fs_r, lo=0.05, hi=2.0, order=2)
    rp = heart.detect_r_peaks(night["ecg"], 200.0) / 200.0
    rp = rp[(rp >= t0) & (rp < t0 + SEG_S)] - t0
    return resp, fs_r, rp


def truth(resp, fs_r, rp):
    rc = respiration.belts_to_resp_channels(resp, fs_r)
    _, _, _, F, names = respiration.analyze(rc)
    centers = np.arange(heart.SUB_S / 2, SEG_S, heart.STEP_S)
    hr = heart.hr_from_rpeaks(rp, centers)
    hF, _ = heart.hr_epoch_features(centers, hr, np.ones_like(hr), len(F))
    return F[:, names.index("rr")], hF[:, 0]


def run(job):
    path, dist, noise, blanket, seed = job
    rng = np.random.default_rng(seed)
    resp, fs_r, rp = segment(path, rng)
    rr_t, hr_t = truth(resp, fs_r, rp)
    fsf = sonar.FRAME_RATE
    k = int(SEG_S * fsf)
    t = np.arange(k) / fsf
    r = np.interp(t, np.arange(len(resp)) / fs_r, resp)
    r = r / (np.percentile(np.abs(r), 90) or 1.0)
    ptp = rng.uniform(2.0, 8.0)
    chest = r * ptp / 2
    heart_mm = np.zeros(k)
    amp = rng.uniform(0.2, 0.4)
    for b in rp + 0.2:
        i0, i1 = int((b - 0.1) * fsf), int((b + 0.4) * fsf)
        tt = t[max(0, i0):min(k, i1)] - b
        heart_mm[max(0, i0):min(k, i1)] += amp * np.exp(-0.5 * (tt / 0.06) ** 2)
    frames, tx = sonar.simulate_frames(chest, dist, k, seed=seed, noise_db=noise,
                                       blanket_db=blanket, heart_mm=heart_mm)
    disp, act, bins = sonar.sonar_resp_and_motion(frames, tx)
    rc = sonar.to_resp_channels(disp)
    _, _, _, F, names = respiration.analyze(rc)
    rr = F[:, names.index("rr")]
    c, hr, q = sonar.heart_from_displacement(disp)
    hF, _ = heart.hr_epoch_features(c, hr, q, len(F))
    n = min(len(rr), len(rr_t))
    er = np.abs(rr[:n] - rr_t[:n])
    eh = np.abs(hF[:n, 0] - hr_t[:n])
    return dict(file=os.path.basename(path), dist=dist, noise=noise, blanket=blanket, ptp=ptp,
                bin_err_cm=float(abs(sonar.dist_of_bin(np.median(bins)) - dist) * 100),
                rr_mae=float(np.nanmean(er)), rr_within1=float(np.nanmean(er <= 1)),
                hr_mae=float(np.nanmean(eh)) if np.isfinite(eh).any() else np.nan,
                hr_within3=float(np.mean(np.where(np.isfinite(eh), eh <= 3, False))),
                hr_cov=float(np.isfinite(eh).mean()))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    files = sorted(glob.glob("/home/user/data/c18/*.npz"))[7::60][:5]
    jobs = [(f, d, nz, bl, 1000 + i) for i, (f, d, nz, bl) in
            enumerate(itertools.product(files, DISTS, NOISE, BLANKET))]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(run, jobs))
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e02_sonar.json")
    json.dump(res, open(out, "w"), indent=1)
    import collections
    agg = collections.defaultdict(list)
    for r in res:
        agg[(r["dist"], r["noise"], r["blanket"])].append(r)
    print(f"{'dist':>5} {'bruit':>6} {'couette':>7} | {'FR err':>6} {'FR±1':>5} | {'FC err':>6} {'FC±3':>5} {'couv':>5}")
    for kk in sorted(agg):
        rs = agg[kk]
        m = lambda key: np.nanmean([r[key] for r in rs])
        print(f"{kk[0]:5.1f} {kk[1]:6d} {kk[2]:7.0f} | {m('rr_mae'):6.2f} {m('rr_within1'):5.0%} | "
              f"{m('hr_mae'):6.2f} {m('hr_within3'):5.0%} {m('hr_cov'):5.0%}")


if __name__ == "__main__":
    main()
