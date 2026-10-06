"""Sonar FMCW sans contact : le haut-parleur de l'iPhone émet des « chirps »
inaudibles (18 → 20,5 kHz), le micro capte leurs échos ; la poitrine qui se soulève
décale l'écho de quelques millimètres → la phase de l'écho tourne (λ ≈ 1,8 cm,
aller-retour : 5 mm de déplacement ≈ 3,5 rad). Même principe que ApneaApp
(Nandakumar et al., Univ. Washington 2015) ou SleepScore.

Chaîne de traitement (identique dans l'app web, web/js/sonar.js) :
 1. trames de N=1024 échantillons alignées sur la période d'émission ;
 2. compression d'impulsion : corrélation circulaire avec le chirp émis, en ne gardant
    que la bande du chirp → profil de distance complexe (1 case ≈ 3,6 mm) ;
 3. ré-alignement sur le trajet direct haut-parleur → micro (pic le plus fort) ;
 4. par époque, choix de la case dont l'énergie dans la bande respiratoire est
    maximale, ajustement de cercle (le fouillis statique décale le centre de l'arc
    décrit dans le plan I/Q) et phase → déplacement en mm ;
 5. l'onde de déplacement alimente la même analyse respiratoire que l'accéléromètre.
"""
import numpy as np

from . import dsp

C_SOUND = 343.0
FS = 48000
N = 1024
F0, F1 = 18000.0, 20500.0
FRAME_RATE = FS / N  # 46,875 Hz
EPOCH_S = 30


def make_chirp(fs=FS, n=N, f0=F0, f1=F1, taper=0.08):
    """Chirp linéaire montant avec bords adoucis (Tukey) pour éviter les clics."""
    t = np.arange(n) / fs
    T = n / fs
    ph = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t * t)
    x = np.sin(ph)
    m = int(taper * n / 2)
    if m > 0:
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / m)
        x[:m] *= w
        x[-m:] *= w[::-1]
    return x


def band_mask(fs=FS, n=N, f0=F0, f1=F1, margin=300.0):
    f = np.fft.fftfreq(n, 1 / fs)
    return (f >= f0 - margin) & (f <= f1 + margin)


DEC = 4                                          # décimation du profil (1 case = 14 mm)
N_KEEP = int(2 * 2.5 / C_SOUND * FS) // DEC      # cases conservées : jusqu'à 2,5 m
MIN_BIN = 10                                     # on ignore < 14 cm (trajet direct)


def range_profiles(frames, tx, fs=FS, dec=DEC):
    """frames (k, N) réels → profils complexes circulaires décimés (k, N/dec) :
    case i ↔ distance i·dec·c/(2·fs). Identique à l'AudioWorklet de l'app web."""
    n = tx.shape[0]
    T = np.fft.fft(tx)
    mask = band_mask(fs, n)
    R = np.fft.fft(frames, axis=1) * np.conj(T)[None, :] * mask[None, :]
    return np.fft.ifft(R, axis=1)[:, ::dec]


def align_direct(prof, n_keep=N_KEEP, smooth=47):
    """Décale chaque trame pour que le trajet direct (pic max) soit en case 0.
    Le pic est médian-filtré sur ~1 s pour ignorer un écho transitoire."""
    k = prof.shape[0]
    peak = np.argmax(np.abs(prof), axis=1)
    h = smooth // 2
    med = np.array([int(np.median(peak[max(0, i - h): i + h + 1])) for i in range(k)])
    idx = (med[:, None] + np.arange(n_keep)[None, :]) % prof.shape[1]
    return np.take_along_axis(prof, idx, axis=1)


def fit_circle(z):
    """Ajustement algébrique (Kåsa) d'un cercle aux points complexes z → centre."""
    x, y = z.real, z.imag
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.mean(z)
    return complex(sol[0] / 2, sol[1] / 2)


def displacement(prof, fs_f=FRAME_RATE, min_bin=MIN_BIN, win_s=60, lam=None):
    """Profils alignés (k, n_bins) → (déplacement mm (k,), case choisie par époque,
    activité par trame)."""
    if lam is None:
        lam = C_SOUND / ((F0 + F1) / 2)
    k, nb = prof.shape
    ne = int(round(EPOCH_S * fs_f))
    n_ep = k // ne
    half = int(win_s * fs_f / 2)
    out = np.zeros(k)
    bins = np.zeros(n_ep, dtype=int)
    # partie dynamique : on retire la moyenne glissante (fouillis statique)
    for e in range(n_ep):
        c = e * ne + ne // 2
        a, b = max(0, c - half), min(k, c + half)
        seg = prof[a:b, min_bin:]
        dyn = seg - seg.mean(axis=0, keepdims=True)
        # énergie dans la bande respiratoire, case par case (FFT lente, zéro-padding)
        nfft = 1 << int(np.ceil(np.log2(max(2, b - a))))
        P = np.abs(np.fft.fft(dyn, n=nfft, axis=0)) ** 2
        f = np.abs(np.fft.fftfreq(nfft, 1 / fs_f))
        band = (f >= 0.1) & (f <= 0.7)
        hi = (f > 0.7) & (f <= 5.0)
        pb = P[band].sum(axis=0)
        ph = P[hi].sum(axis=0) + 1e-30
        score = pb * pb / (pb + ph)  # énergie respiratoire, pénalisée si large bande
        j = int(np.argmax(score))
        bins[e] = j + min_bin
        z = seg[:, j]
        cen = fit_circle(z)
        # si l'arc est minuscule, le cercle est mal défini → centre = moyenne
        if not np.isfinite(cen) or abs(cen - z.mean()) > 50 * np.std(np.abs(z - z.mean())) + 1e-12:
            cen = z.mean()
        ang = np.unwrap(np.angle(z - cen))
        d = ang * lam / (4 * np.pi) * 1000.0  # mm
        lo, hi_ = e * ne - a, e * ne - a + ne
        piece = d[lo:hi_]
        out[e * ne:(e + 1) * ne] = piece - piece.mean()
    # activité : variation globale du profil d'une trame à la suivante (mouvements)
    act = np.zeros(k)
    act[1:] = np.sqrt((np.abs(np.diff(prof[:, min_bin:], axis=0)) ** 2).sum(axis=1))
    return out, bins, act


def simulate_frames(chest_mm, dist_m, n_frames, seed=0, fs=FS, refl=0.3, blanket_db=6.0,
                    noise_db=-75.0, n_clutter=12, heart_mm=None, extra=None):
    """Simule les trames reçues (k, N) dans le domaine fréquentiel (exact pour un
    train périodique de chirps) : trajet direct + fouillis statique + écho thoracique
    dont la distance varie avec chest_mm(t) (mm, une valeur par trame).

    Niveaux relatifs au trajet direct (amplitude 1) :
      écho ∝ refl·(0,05/d)²·10^(−blanket_db/20)    (≈ −50 dB à 0,5 m, −62 dB à 1 m)
      bruit micro : noise_db dB par échantillon par rapport au trajet direct.
    """
    rng = np.random.default_rng(seed)
    tx = make_chirp(fs)
    TX = np.fft.fft(tx)
    f = np.fft.fftfreq(N, 1 / fs)
    latency = rng.integers(100, 600)  # latence E/S inconnue (échantillons)
    # réponse HP+micro : chute de ~6 dB entre 18 et 20,5 kHz
    H = np.where(np.abs(f) > 0, 10 ** (-6 / 20 * np.clip((np.abs(f) - F0) / (F1 - F0), 0, 1)), 0)
    paths = [(0.03, 1.0)]  # trajet direct ~3 cm
    for _ in range(n_clutter):
        d = rng.uniform(0.2, 3.0)
        paths.append((d, rng.uniform(0.2, 1.0) * (0.05 / d) ** 2))
    static = np.zeros(N, dtype=complex)
    for d, a in paths:
        static += a * np.exp(-2j * np.pi * f * (2 * d / C_SOUND + latency / fs))
    static *= TX * H
    a_ch = refl * (0.05 / dist_m) ** 2 * 10 ** (-blanket_db / 20)
    disp = np.asarray(chest_mm, dtype=np.float64)[:n_frames] / 1000.0
    if heart_mm is not None:
        disp = disp + np.asarray(heart_mm)[:n_frames] / 1000.0
    sigma = 10 ** (noise_db / 20)
    frames = np.empty((n_frames, N))
    for k in range(n_frames):
        tau = 2 * (dist_m + disp[k]) / C_SOUND + latency / fs
        X = static + a_ch * TX * H * np.exp(-2j * np.pi * f * tau)
        if extra is not None:
            X = X + extra(k, f, TX * H, latency / fs)
        frames[k] = np.fft.ifft(X).real + rng.normal(0, sigma, N)
    return frames, tx


def sonar_resp_and_motion(frames, tx):
    """Trames → (onde de déplacement mm à FRAME_RATE, activité/trame, cases)."""
    prof = align_direct(range_profiles(frames, tx))
    disp, bins, act = displacement(prof)
    return disp, act, bins


def to_resp_channels(disp, fs_f=FRAME_RATE):
    from . import respiration
    return respiration.belts_to_resp_channels(disp, fs_f)


def heart_from_displacement(disp, fs_f=FRAME_RATE):
    """FC depuis le déplacement thoracique (composante 0,8–3 Hz), via le même suivi
    par autocorrélation que le BCG."""
    from . import heart
    x = dsp.filtfilt(disp, fs_f, lo=0.8, hi=3.0, order=2)
    x = dsp.resample_linear(x, fs_f, heart.FS_ENV)
    acf, _, centers, lag_min = heart.windowed_acf(x[:, None])
    hr, q = heart.track_hr(acf, lag_min)
    return centers, hr, q


def dist_of_bin(b, fs=FS, dec=DEC):
    return b * dec * C_SOUND / (2 * fs)

