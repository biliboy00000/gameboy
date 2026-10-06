"""Primitives de traitement du signal.

Tout est volontairement écrit avec des briques simples (biquads de Butterworth du
« RBJ cookbook », filtrage aller-retour avec réflexion impaire, FFT) pour pouvoir être
reproduit à l'identique en JavaScript dans l'application iPhone (web/js/dsp.js).
Le test tests/test_parity.py vérifie l'égalité numérique Python ⇄ JS.
"""
import math

import numpy as np
import scipy.signal

# Facteurs Q des sections d'un Butterworth d'ordre 2 et 4
_Q = {2: [1 / math.sqrt(2)], 4: [0.5411961001461970, 1.3065629648763766]}


def biquad(kind, fc, fs, q):
    """Coefficients normalisés [b0, b1, b2, 1, a1, a2] (format SOS de scipy)."""
    w0 = 2 * math.pi * fc / fs
    c, s = math.cos(w0), math.sin(w0)
    alpha = s / (2 * q)
    if kind == "lp":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
    elif kind == "hp":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    else:
        raise ValueError(kind)
    a0 = 1 + alpha
    return [b[0] / a0, b[1] / a0, b[2] / a0, 1.0, -2 * c / a0, (1 - alpha) / a0]


def butter_sos(fs, lo=None, hi=None, order=4):
    """Passe-haut (lo), passe-bas (hi) ou passe-bande (les deux), Butterworth."""
    sos = []
    if lo is not None:
        sos += [biquad("hp", lo, fs, q) for q in _Q[order]]
    if hi is not None:
        sos += [biquad("lp", hi, fs, q) for q in _Q[order]]
    return np.array(sos)


def filtfilt(x, fs, lo=None, hi=None, order=4, axis=0):
    """Filtrage à phase nulle : réflexion impaire, aller, retour (état initial nul)."""
    x = np.asarray(x, dtype=np.float64)
    sos = butter_sos(fs, lo, hi, order)
    fmin = min(f for f in (lo, hi) if f is not None)
    n = x.shape[axis]
    pad = int(min(n - 1, round(3 * fs / fmin)))
    x = np.moveaxis(x, axis, 0)
    if pad > 0:
        head = 2 * x[0] - x[pad:0:-1]
        tail = 2 * x[-1] - x[-2:-pad - 2:-1]
        x = np.concatenate([head, x, tail], axis=0)
    y = scipy.signal.sosfilt(sos, x, axis=0)
    y = scipy.signal.sosfilt(sos, y[::-1], axis=0)[::-1]
    if pad > 0:
        y = y[pad:-pad]
    return np.moveaxis(y, 0, axis)


def resample_linear(x, fs_in, fs_out, axis=0):
    """Ré-échantillonnage par interpolation linéaire sur une grille régulière."""
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, 0)
    n = x.shape[0]
    t_out = np.arange(int(math.floor((n - 1) * fs_out / fs_in)) + 1) / fs_out
    pos = t_out * fs_in
    i0 = np.minimum(np.floor(pos).astype(int), n - 2)
    fr = (pos - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    y = x[i0] * (1 - fr) + x[i0 + 1] * fr
    return np.moveaxis(y, 0, axis)


def resample_timestamps(t, x, fs):
    """Signal irrégulier (horodatages en s, ex. événements devicemotion de Safari)
    → grille régulière à fs Hz par interpolation linéaire, canal par canal."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    grid = t[0] + np.arange(int((t[-1] - t[0]) * fs) + 1) / fs
    if x.ndim == 1:
        return grid, np.interp(grid, t, x)
    return grid, np.stack([np.interp(grid, t, x[:, k]) for k in range(x.shape[1])], axis=1)


def epochs_view(x, n_per_epoch):
    """Découpe (n,) ou (n, c) en (n_epochs, n_per_epoch[, c])."""
    n_ep = x.shape[0] // n_per_epoch
    return x[: n_ep * n_per_epoch].reshape((n_ep, n_per_epoch) + x.shape[1:])


def window(x, center, half):
    """Fenêtre [center-half, center+half) bornée au signal."""
    a = max(0, center - half)
    b = min(x.shape[0], center + half)
    return x[a:b]


def autocorr(x, max_lag):
    """Autocorrélation normalisée (biaisée) pour les décalages 0..max_lag."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    r = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1]
    return r / r[0] if r[0] > 0 else np.zeros(max_lag + 1)


def parabolic_peak(y, i):
    """Raffinement sub-échantillon d'un maximum local."""
    if 0 < i < len(y) - 1:
        d = y[i - 1] - 2 * y[i] + y[i + 1]
        if d < 0:
            return i + 0.5 * (y[i - 1] - y[i + 1]) / d
    return float(i)


def hann_spectrum(x, fs, nfft):
    """Périodogramme (fenêtre de Hann, zéro-padding) → (fréquences, puissance)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    w = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(len(x)) / max(1, len(x) - 1))
    p = np.abs(np.fft.rfft(x * w, nfft)) ** 2
    return np.arange(len(p)) * fs / nfft, p


def robust_scale(x):
    """Échelle robuste (MAD → σ), jamais nulle."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    mad = np.median(np.abs(x - np.median(x)))
    return float(mad * 1.4826) if mad > 0 else float(np.std(x) or 1.0)
