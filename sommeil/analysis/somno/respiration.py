"""Respiration : du capteur brut à des caractéristiques par époque de 30 s.

Un iPhone posé sur le matelas « voit » la respiration surtout comme une petite
inclinaison périodique du matelas (0,05–0,5°) : la gravité se projette différemment
sur les axes x/y de l'accéléromètre (≈ 1–9 mg), et le gyroscope mesure la vitesse de
cette inclinaison. On intègre le gyroscope pour obtenir un angle (en phase avec
l'accéléromètre, même unité m/s² via g·θ), puis une ACP glissante choisit la meilleure
combinaison des 6 canaux. Le même code traite les ceintures thoraco-abdominales d'une
polysomnographie (validation) ou l'onde respiratoire du sonar.
"""
import numpy as np

from . import dsp

FS_R = 10.0          # fréquence de travail de la respiration (Hz)
BAND = (0.08, 1.0)   # 4,8 à 60 cycles/min
EPOCH_S = 30
G = 9.80665


def motion_to_resp_channels(acc, gyro, fs):
    """acc (n,3) m/s² gravité incluse, gyro (n,3) rad/s → (m,6) à FS_R, en m/s² équivalents."""
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    # gyro → angle d'inclinaison (rad), dérive supprimée par le passe-bande ensuite
    ang = np.cumsum(dsp.filtfilt(gyro, fs, lo=0.03, order=2), axis=0) / fs
    x = np.concatenate([acc, G * ang], axis=1)
    x = dsp.filtfilt(x, fs, hi=2.0, order=4)
    x = dsp.resample_linear(x, fs, FS_R)
    x = dsp.filtfilt(x, FS_R, lo=BAND[0], hi=BAND[1], order=2)
    # blanchiment : chaque canal divisé par son niveau de bruit (échelle robuste de la
    # dérivée). Sans cela l'ACP privilégie les canaux les plus bruités ; avec, elle
    # maximise le rapport signal/bruit (gyroscope intégré souvent 3× meilleur).
    sc = np.array([dsp.robust_scale(np.diff(x[:, j])) for j in range(x.shape[1])])
    return x / np.where(sc > 0, sc, 1.0)


def belts_to_resp_channels(belts, fs):
    """Ceintures PSG (n,c) → (m,c) à FS_R, passe-bande respiratoire."""
    x = np.asarray(belts, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = dsp.filtfilt(x, fs, hi=2.0, order=4)
    x = dsp.resample_linear(x, fs, FS_R)
    return dsp.filtfilt(x, FS_R, lo=BAND[0], hi=BAND[1], order=2)


def pca_waveform(chans, win_s=120):
    """Projette les canaux sur leur 1re composante principale, recalculée à chaque
    époque sur une fenêtre de win_s centrée ; le signe est aligné d'une époque à
    l'autre pour garder une onde continue. → (onde (m,), poids (n_ep, c))."""
    n, c = chans.shape
    ne = int(EPOCH_S * FS_R)
    n_ep = n // ne
    half = int(win_s * FS_R / 2)
    out = np.zeros(n_ep * ne)
    weights = np.zeros((n_ep, c))
    prev = None
    # énergie par seconde : les secondes agitées (mouvements) sont exclues de l'ACP,
    # sinon la composante principale suit le mouvement au lieu de la respiration
    sec = int(FS_R)
    n_s = n // sec
    e_s = (chans[: n_s * sec] ** 2).sum(axis=1).reshape(n_s, sec).mean(axis=1)
    for e in range(n_ep):
        c0 = e * ne + ne // 2
        a, b = max(0, c0 - half), min(n, c0 + half)
        sa, sb = a // sec, min(n_s, b // sec)
        es = e_s[sa:sb]
        keep = es <= 4 * np.median(es) if len(es) else es
        rows = np.repeat(keep, sec)
        seg = chans[sa * sec: sb * sec][rows]
        if len(seg) < 10 * sec:
            seg = chans[a:b]
        cov = seg.T @ seg / max(1, len(seg))
        vals, vecs = np.linalg.eigh(cov)
        w = vecs[:, -1]
        if prev is not None and np.dot(w, prev) < 0:
            w = -w
        elif prev is None and w[np.argmax(np.abs(w))] < 0:
            w = -w
        prev = w
        weights[e] = w
        out[e * ne:(e + 1) * ne] = chans[e * ne:(e + 1) * ne] @ w
    return out, weights


def detect_breaths(w, fs=FS_R, win_s=60):
    """Détection respiration par passage par zéro avec hystérésis.
    Seuil h = 0,15 × amplitude robuste locale. → (instants en s, amplitudes)."""
    n = len(w)
    blk = int(10 * fs)
    nb = max(1, n // blk)
    # amplitude robuste par blocs de 10 s, lissée sur win_s
    loc = np.array([np.percentile(np.abs(w[i * blk:(i + 1) * blk]), 90) if i * blk < n else 0
                    for i in range(nb)])
    k = max(1, int(win_s / 10))
    loc_s = np.array([np.median(loc[max(0, i - k // 2): i + k // 2 + 1]) for i in range(nb)])
    h = 0.15 * np.repeat(loc_s, blk)
    if len(h) < n:
        h = np.concatenate([h, np.full(n - len(h), h[-1] if len(h) else 0.0)])
    times = []
    state = 0  # -1 sous -h, +1 au-dessus de +h
    last_zc = None
    for i in range(1, n):
        x = w[i]
        if w[i - 1] < 0 <= x:
            last_zc = (i - 1) + (-w[i - 1]) / (x - w[i - 1])
        if x < -h[i]:
            state = -1
        elif x > h[i]:
            if state == -1 and last_zc is not None:
                t = last_zc / fs
                if not times or t - times[-1] >= 1.0:
                    times.append(t)
            state = 1
    times = np.array(times)
    # amplitude crête-à-crête de chaque cycle (entre deux débuts d'inspiration)
    amps = np.full(len(times), np.nan)
    idx = np.round(times * fs).astype(int)
    for k in range(len(idx) - 1):
        seg = w[idx[k]:idx[k + 1] + 1]
        amps[k] = seg.max() - seg.min()
    return times, amps


def epoch_features(w, breath_t, breath_a, fs=FS_R, ctx_s=60):
    """Caractéristiques respiratoires par époque, calculées sur une fenêtre de ctx_s
    centrée sur l'époque (60 s → ~15 respirations, assez pour une variabilité)."""
    ne = int(EPOCH_S * fs)
    n_ep = len(w) // ne
    half = int(ctx_s * fs / 2)
    nfft = 1024
    names = ["rr", "rr_spec", "ibi_cv", "amp_cv", "reg", "spec_conc", "amp", "n_breaths"]
    F = np.full((n_ep, len(names)), np.nan)
    for e in range(n_ep):
        c = e * ne + ne // 2
        seg = dsp.window(w, c, half)
        t0, t1 = (c - half) / fs, (c + half) / fs
        f, p = dsp.hann_spectrum(seg, fs, nfft)
        band = (f >= BAND[0]) & (f <= 0.7)
        if p[band].sum() > 0:
            i = int(np.argmax(np.where(band, p, -1)))
            fi = dsp.parabolic_peak(p, i) * fs / nfft
            F[e, 1] = 60 * fi
            near = band & (np.abs(f - f[i]) <= 0.04)
            F[e, 5] = p[near].sum() / p[band].sum()
            lag = int(round(fs / max(fi, 1e-3)))
            if lag < len(seg) - 1:
                F[e, 4] = dsp.autocorr(seg, lag)[lag]
        F[e, 6] = np.std(w[e * ne:(e + 1) * ne])
        sel = (breath_t >= t0) & (breath_t < t1)
        bt = breath_t[sel]
        F[e, 7] = len(bt)
        if len(bt) >= 3:
            ibi = np.diff(bt)
            F[e, 0] = 60 / np.median(ibi)
            F[e, 2] = np.std(ibi) / np.mean(ibi)
            ba = breath_a[sel]
            ba = ba[np.isfinite(ba)]
            if len(ba) >= 3 and np.mean(ba) > 0:
                F[e, 3] = np.std(ba) / np.mean(ba)
    return F, names


def analyze(chans):
    """Canaux respiratoires (m,c) à FS_R → (onde, instants, amplitudes, features, noms)."""
    w, _ = pca_waveform(chans)
    bt, ba = detect_breaths(w)
    F, names = epoch_features(w, bt, ba)
    return w, bt, ba, F, names
