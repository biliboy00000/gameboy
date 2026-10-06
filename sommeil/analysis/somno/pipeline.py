"""Chaînes de traitement complètes.

  phone_features(acc, gyro, fs)   iPhone sur le matelas  → caractéristiques par époque
  psg_features(night)             PSG « idéale » (ceintures + ECG) → mêmes caractéristiques
  hybrid_features(night, ...)     PSG → signaux iPhone simulés → chaîne iPhone
"""
import numpy as np

from . import dsp, features, heart, movement, respiration, simulate

EPOCH_S = 30


def phone_features(acc, gyro, fs=50.0, return_signals=False):
    """acc (n,3) m/s² gravité incluse, gyro (n,3) rad/s, échantillonnés à fs."""
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    n_ep = int(len(acc) / fs // EPOCH_S)
    rc = respiration.motion_to_resp_channels(acc, gyro, fs)
    w, bt, ba, rF, _ = respiration.analyze(rc)
    motion6 = np.concatenate([acc, gyro * 9.80665 / 10], axis=1)  # gyro ≈ même échelle
    centers, hr, q = heart.bcg_heart_rate(motion6, fs)
    hF, _ = heart.hr_epoch_features(centers, hr, q, n_ep)
    act = movement.relative_activity(movement.activity_per_second(acc, fs),
                                     movement.activity_per_second(gyro, fs))
    mF, _, moving = movement.epoch_features(act, n_ep)
    X, names = features.assemble(rF[:n_ep], hF, mF)
    if return_signals:
        return X, names, dict(resp_wave=w, breaths=bt, breath_amp=ba, hr_t=centers, hr=hr,
                              hr_q=q, activity=act, moving=moving)
    return X, names


def psg_movement(night):
    """Mouvements visibles sur une PSG : composante > 2 Hz des ceintures, rapportée à
    l'amplitude respiratoire locale (médiane glissante 1 min). Un artefact de
    mouvement est grand devant la respiration ; les harmoniques respiratoires, non.
    L'ECG n'est volontairement pas utilisé : son bruit haute fréquence trahit le
    tonus musculaire (EMG), information qu'un téléphone n'a pas. Seuil 0,3 calibré
    sur 25 nuits (proxy conservateur : ≈ 20 % des époques d'éveil, < 5 % en N2-N3),
    ramené à l'échelle de movement.THR."""
    fs_r = float(night["fs_resp"])
    belts = np.stack([night["chest"], night["abd"]], axis=1).astype(np.float64)
    hf = movement.activity_per_second(belts, fs_r, lo=2.0)
    rb = dsp.filtfilt(belts, fs_r, lo=0.1, hi=1.0, order=2)
    br = movement.activity_per_second(rb, fs_r, lo=None)
    n = min(len(hf), len(br))
    k = 30
    brs = np.array([np.median(br[max(0, i - k): i + k + 1]) for i in range(n)])
    return hf[:n] / np.maximum(brs, 1e-9) * (movement.THR / 0.3)


def psg_features(night, rpeaks_s=None):
    """Caractéristiques « idéales » depuis la PSG (borne haute de ce que le téléphone
    peut espérer : respiration parfaite, battements exacts)."""
    fs_r = float(night["fs_resp"])
    n_ep = len(night["stages"])
    belts = np.stack([night["chest"], night["abd"]], axis=1)
    rc = respiration.belts_to_resp_channels(belts, fs_r)
    w, bt, ba, rF, _ = respiration.analyze(rc)
    if rpeaks_s is None:
        rpeaks_s = heart.detect_r_peaks(night["ecg"], float(night["fs_ecg"])) / float(night["fs_ecg"])
    centers = np.arange(heart.SUB_S / 2, n_ep * EPOCH_S, heart.STEP_S)
    hr = heart.hr_from_rpeaks(rpeaks_s, centers)
    hF, _ = heart.hr_epoch_features(centers, hr, np.ones_like(hr), n_ep)
    act = psg_movement(night)
    mF, _, _ = movement.epoch_features(act, n_ep)
    n = min(len(rF), n_ep)
    return features.assemble(rF[:n], hF[:n], mF[:n])


def hybrid_signals(night, rpeaks_s, seed=0, **kw):
    """Signaux iPhone simulés à partir d'une vraie nuit PSG."""
    fs_r = float(night["fs_resp"])
    resp = np.asarray(night["chest"], dtype=np.float64) + np.asarray(night["abd"], dtype=np.float64)
    act = psg_movement(night)
    moving = act > movement.THR
    moving = np.convolve(moving.astype(float), np.ones(3), mode="same") > 0
    n_ep = len(night["stages"])
    beats = rpeaks_s + 0.22 + np.random.default_rng(seed).normal(0, 0.008, size=len(rpeaks_s))
    return simulate.synth_phone(resp, beats, moving, fs=50.0, seed=seed,
                                duration_s=n_ep * EPOCH_S, fs_resp=fs_r, **kw)


def hybrid_features(night, rpeaks_s=None, seed=0, **kw):
    if rpeaks_s is None:
        rpeaks_s = heart.detect_r_peaks(night["ecg"], float(night["fs_ecg"])) / float(night["fs_ecg"])
    acc, gyro, info = hybrid_signals(night, rpeaks_s, seed=seed, **kw)
    X, names = phone_features(acc, gyro, 50.0)
    return X, names, info
