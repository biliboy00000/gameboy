"""Cœur : fréquence cardiaque par ballistocardiographie (BCG) et détection R sur ECG.

BCG : chaque éjection ventriculaire fait « reculer » le corps et le matelas ; un
iPhone posé dessus enregistre une petite salve vibratoire (4–12 Hz) à chaque
battement, sur l'accéléromètre comme sur le gyroscope. On n'essaie pas de repérer
chaque pic (trop fragile sur un matelas) : on calcule l'enveloppe d'énergie de
chaque canal, son autocorrélation sur des fenêtres de 10 s, et on suit la période
cardiaque au cours de la nuit par programmation dynamique (Viterbi) — ce qui
élimine les sauts d'octave et les fenêtres bruitées.
"""
import numpy as np

from . import dsp

FS_ENV = 25.0
SUB_S = 10.0     # fenêtre d'autocorrélation
STEP_S = 5.0     # pas → une estimation toutes les 5 s
HR_MIN, HR_MAX = 38.0, 150.0


def bcg_envelopes(motion, fs):
    """motion (n,c) : canaux normalisés (acc en m/s², gyro en rad/s) → enveloppes (m,c)."""
    x = dsp.filtfilt(motion, fs, lo=4.0, hi=min(12.0, 0.45 * fs), order=2)
    e = dsp.filtfilt(x * x, fs, hi=2.5, order=2)
    e = dsp.resample_linear(e, fs, FS_ENV)
    med = np.median(np.abs(e), axis=0)
    return e / np.where(med > 0, med, 1.0)


def windowed_acf(env, fs=FS_ENV):
    """Autocorrélations moyennes (pondérées par la qualité de chaque canal) sur
    des fenêtres glissantes. → (acf (n_win, n_lag), lags (s), centres (s))."""
    if env.ndim == 1:
        env = env[:, None]
    n, c = env.shape
    lw, ls = int(SUB_S * fs), int(STEP_S * fs)
    lag_min, lag_max = int(np.floor(fs * 60 / HR_MAX)), int(np.ceil(fs * 60 / HR_MIN))
    starts = np.arange(0, max(1, n - lw + 1), ls)
    out = np.zeros((len(starts), lag_max + 1))
    for k, s in enumerate(starts):
        acc, wsum = np.zeros(lag_max + 1), 0.0
        for j in range(c):
            seg = env[s:s + lw, j]
            if len(seg) <= lag_max + 1 or np.std(seg) == 0:
                continue
            r = dsp.autocorr(seg, lag_max)
            q = max(0.0, r[lag_min:].max())
            acc += q * q * r
            wsum += q * q
        if wsum > 0:
            out[k] = acc / wsum
    lags = np.arange(lag_max + 1) / fs
    centers = (starts + lw / 2) / fs
    return out, lags, centers, lag_min


def track_hr(acf, lag_min, fs=FS_ENV, penalty=0.06):
    """Viterbi sur la grille des décalages : maximise Σ acf − penalty·|Δdécalage|.
    → (fréquence cardiaque bpm, qualité = valeur d'autocorrélation au point retenu)."""
    sc = acf[:, lag_min:]
    nw, nl = sc.shape
    if nw == 0:
        return np.zeros(0), np.zeros(0)
    lag_idx = np.arange(nl)
    trans = -penalty * np.abs(lag_idx[:, None] - lag_idx[None, :])  # [précédent, courant]
    D = sc[0].copy()
    back = np.zeros((nw, nl), dtype=np.int32)
    for t in range(1, nw):
        m = D[:, None] + trans
        back[t] = np.argmax(m, axis=0)
        D = m[back[t], lag_idx] + sc[t]
    path = np.zeros(nw, dtype=np.int32)
    path[-1] = int(np.argmax(D))
    for t in range(nw - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    hr = np.zeros(nw)
    q = np.zeros(nw)
    for t in range(nw):
        i = path[t] + lag_min
        lag = dsp.parabolic_peak(acf[t], i) if 0 < i < acf.shape[1] - 1 else i
        lag = min(max(lag, i - 0.5), i + 0.5)
        hr[t] = 60 * fs / lag
        q[t] = acf[t, i]
    return hr, q


def bcg_heart_rate(motion, fs):
    """Signal de mouvement (n,c) → (instants s, FC bpm, qualité)."""
    env = bcg_envelopes(motion, fs)
    acf, _, centers, lag_min = windowed_acf(env)
    hr, q = track_hr(acf, lag_min)
    return centers, hr, q


# ---------------------------------------------------------------- ECG (validation)
def detect_r_peaks(ecg, fs):
    """Détecteur R de type Pan-Tompkins, polarité indifférente. → indices."""
    x = dsp.filtfilt(np.asarray(ecg, dtype=np.float64), fs, lo=5.0, hi=20.0, order=2)
    d = np.gradient(x) ** 2
    k = max(1, int(0.12 * fs))
    integ = np.convolve(d, np.ones(k) / k, mode="same")
    # seuil adaptatif : 35 % du 98e centile sur des blocs de 10 s
    blk = int(10 * fs)
    thr = np.zeros_like(integ)
    for s in range(0, len(integ), blk):
        thr[s:s + blk] = 0.35 * np.percentile(integ[s:s + 2 * blk], 98)
    above = integ > thr
    edges = np.flatnonzero(np.diff(above.astype(np.int8)) == 1) + 1
    refr = int(0.25 * fs)
    peaks, last = [], -refr
    ax = np.abs(x)
    for s in edges:
        a, b = max(0, s - int(0.1 * fs)), min(len(x), s + int(0.15 * fs))
        p = a + int(np.argmax(ax[a:b]))
        if p - last >= refr:
            peaks.append(p)
            last = p
    return np.array(peaks, dtype=int)


def hr_from_rpeaks(rpeaks_s, centers):
    """FC moyenne sur des fenêtres de SUB_S centrées (comme le BCG), qualité 1."""
    hr = np.full(len(centers), np.nan)
    if len(rpeaks_s) < 3:
        return hr
    rr = np.diff(rpeaks_s)
    tm = rpeaks_s[1:]
    ok = (rr > 60 / 200) & (rr < 60 / 30)
    rr, tm = rr[ok], tm[ok]
    for k, c in enumerate(centers):
        sel = (tm >= c - SUB_S / 2) & (tm < c + SUB_S / 2)
        if sel.sum() >= 3:
            hr[k] = 60 / np.median(rr[sel])
    return hr


def hr_epoch_features(centers, hr, q, n_ep, epoch_s=30, q_min=0.25, ctx_s=60):
    """Par époque : FC médiane, écart-type de la FC sur la fenêtre (variabilité lente),
    qualité moyenne. Fenêtres de qualité < q_min ignorées."""
    names = ["hr", "hr_std", "hr_q"]
    F = np.full((n_ep, 3), np.nan)
    good = np.isfinite(hr) & (q >= q_min)
    for e in range(n_ep):
        c = (e + 0.5) * epoch_s
        sel = (np.abs(centers - c) <= ctx_s / 2)
        F[e, 2] = np.mean(q[sel]) if sel.any() else 0.0
        s2 = sel & good
        if s2.sum() >= 3:
            F[e, 0] = np.median(hr[s2])
            F[e, 1] = np.std(hr[s2])
    return F, names
