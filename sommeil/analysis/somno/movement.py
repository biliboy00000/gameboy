"""Mouvements corporels (actimétrie de lit).

Un mouvement de la personne secoue le matelas cent à mille fois plus fort que la
respiration : on mesure, seconde par seconde, l'énergie au-dessus de 1 Hz de
l'accéléromètre et du gyroscope, rapportée au niveau de repos de la nuit (bruit du
capteur + BCG). Les changements de position (retournements) se voient en plus comme
une rotation durable du vecteur gravité.
"""
import numpy as np

from . import dsp

EPOCH_S = 30
THR = 4.0  # seuil d'activité, en multiples du niveau de repos


def activity_per_second(x, fs, lo=1.0):
    """x (n,c) → énergie RMS (somme des canaux) par seconde au-dessus de lo Hz."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    hp = dsp.filtfilt(x, fs, lo=lo, order=2) if lo else x
    e = (hp * hp).sum(axis=1)
    fs = int(round(fs))
    n = len(e) // fs
    return np.sqrt(e[: n * fs].reshape(n, fs).mean(axis=1))


def relative_activity(*acts):
    """Combine plusieurs activités par seconde, chacune rapportée à son niveau de
    repos (10e centile de la nuit). → activité relative (max des sources)."""
    n = min(len(a) for a in acts)
    rel = np.zeros(n)
    for a in acts:
        a = np.asarray(a[:n], dtype=np.float64)
        base = np.percentile(a, 10)
        rel = np.maximum(rel, a / (base if base > 0 else 1.0))
    return rel


def posture_changes(acc, fs, moving_s, min_deg=20.0):
    """Retournements : angle entre la gravité moyenne avant et après chaque épisode
    de mouvement. → liste (seconde, angle°)."""
    acc = np.asarray(acc, dtype=np.float64)
    n = len(moving_s)
    g = dsp.epochs_view(acc[: n * int(fs)], int(fs)).mean(axis=1)
    out = []
    i = 0
    while i < n:
        if moving_s[i]:
            j = i
            while j < n and moving_s[j]:
                j += 1
            a = g[max(0, i - 5):i]
            b = g[j:min(n, j + 5)]
            if len(a) and len(b):
                u, v = a.mean(axis=0), b.mean(axis=0)
                cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
                ang = float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
                if ang >= min_deg:
                    out.append((i, ang))
            i = j
        else:
            i += 1
    return out


def epoch_features(rel, n_ep, thr=THR):
    """Par époque : fraction de secondes en mouvement, intensité (log), temps depuis
    le dernier mouvement et jusqu'au prochain (log minutes)."""
    names = ["mov_frac", "mov_int", "t_since_mov", "t_to_mov"]
    n = n_ep * EPOCH_S
    rel = np.concatenate([rel[:n], np.ones(max(0, n - len(rel)))])
    moving = rel > thr
    ep = moving.reshape(n_ep, EPOCH_S)
    F = np.zeros((n_ep, 4))
    F[:, 0] = ep.mean(axis=1)
    F[:, 1] = np.log1p(np.maximum(rel.reshape(n_ep, EPOCH_S) - thr, 0).sum(axis=1))
    any_mov = ep.any(axis=1)
    big = 120.0  # minutes, plafond
    last = -1e9
    for e in range(n_ep):
        if any_mov[e]:
            last = e
        F[e, 2] = np.log1p(min(big, (e - last) * EPOCH_S / 60))
    nxt = 1e9
    for e in range(n_ep - 1, -1, -1):
        if any_mov[e]:
            nxt = e
        F[e, 3] = np.log1p(min(big, (nxt - e) * EPOCH_S / 60))
    return F, names, moving
