"""Télécharge un sous-ensemble de la base PhysioNet/CinC Challenge 2018 (PSG complètes,
994 nuits annotées AASM) depuis le miroir S3 public de PhysioNet, et ne garde que ce
qu'un iPhone posé sur le matelas peut « voir » :

  - ceintures respiratoires thorax / abdomen  (≈ signal de respiration du téléphone)
  - ECG                                        (≈ battements BCG du téléphone)
  - débit d'air + SaO2                          (pour valider la détection d'apnées)
  - hypnogramme par époque de 30 s + événements respiratoires

Chaque nuit est réduite à ~15 Mo (.npz) puis les fichiers bruts (~135 Mo) sont effacés.

    python fetch_challenge2018.py --n 200 --out /home/user/data/c18
"""
import argparse
import concurrent.futures as cf
import os
import subprocess
import tempfile

import numpy as np
import scipy.io
import scipy.signal
import wfdb

BASE = "https://physionet-open.s3.amazonaws.com/challenge-2018/1.0.0/training"
STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}  # -1 = non scoré
FS = 200
EPOCH = 30


def curl(url, dst):
    subprocess.run(["curl", "-sS", "--retry", "4", "-o", dst, url], check=True)


def convert(rec, out_dir):
    dst = os.path.join(out_dir, rec + ".npz")
    if os.path.exists(dst):
        return rec, "exists"
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        for ext in (".hea", ".mat", ".arousal"):
            curl(f"{BASE}/{rec}/{rec}{ext}", os.path.join(tmp, rec + ext))
        hdr = wfdb.rdheader(os.path.join(tmp, rec))
        names = hdr.sig_name
        val = scipy.io.loadmat(os.path.join(tmp, rec + ".mat"))["val"]
        n = val.shape[1]

        def phys(name):
            i = names.index(name)
            return (val[i].astype(np.float64) - hdr.baseline[i]) / hdr.adc_gain[i]

        def to25(x):
            return scipy.signal.decimate(x, 8, ftype="fir", zero_phase=True).astype(np.float32)

        ann = wfdb.rdann(os.path.join(tmp, rec), "arousal")
        n_ep = n // (FS * EPOCH)
        stages = np.full(n_ep, -1, dtype=np.int8)
        trans = [(s, STAGE_CODES[a]) for s, a in zip(ann.sample, ann.aux_note) if a in STAGE_CODES]
        for k, (s, code) in enumerate(trans):
            e = trans[k + 1][0] if k + 1 < len(trans) else n
            stages[s // (FS * EPOCH): e // (FS * EPOCH)] = code
        events, open_ev = [], {}
        for s, a in zip(ann.sample, ann.aux_note):
            if a.startswith("(resp_"):
                open_ev[a[1:]] = s
            elif a.startswith("resp_") and a.endswith(")") and a[:-1] in open_ev:
                events.append((open_ev.pop(a[:-1]) / FS, s / FS, a[5:-1]))
        sao2 = phys("SaO2")
        np.savez_compressed(
            dst,
            fs_resp=25, fs_ecg=FS, epoch=EPOCH,
            chest=to25(phys("CHEST")), abd=to25(phys("ABD")), airflow=to25(phys("AIRFLOW")),
            sao2=sao2[: n - n % FS].reshape(-1, FS).mean(axis=1).astype(np.float32),
            ecg=np.clip(phys("ECG") * 1000, -32768, 32767).astype(np.int16),  # µV
            stages=stages,
            ev_start=np.array([e[0] for e in events]), ev_end=np.array([e[1] for e in events]),
            ev_type=np.array([e[2] for e in events]),
        )
    return rec, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="/home/user/data/c18")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs_file = os.path.join(a.out, "RECORDS")
    if not os.path.exists(recs_file):
        curl(f"{BASE}/RECORDS", recs_file)
    recs = [r.strip().strip("/") for r in open(recs_file) if r.strip()]
    # sous-échantillonnage déterministe et réparti sur toute la base
    recs = recs[:: max(1, len(recs) // a.n)][: a.n]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(convert, r, a.out) for r in recs]
        for i, f in enumerate(cf.as_completed(futs)):
            try:
                rec, st = f.result()
                print(f"[{i + 1}/{len(recs)}] {rec} {st}", flush=True)
            except Exception as e:  # une nuit corrompue ne doit pas tout arrêter
                print(f"[{i + 1}/{len(recs)}] ERREUR {e!r}", flush=True)


if __name__ == "__main__":
    main()
