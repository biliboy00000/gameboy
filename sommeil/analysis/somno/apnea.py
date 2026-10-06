"""Événements respiratoires (apnées / hypopnées) à partir de l'onde respiratoire.

Critère inspiré de l'AASM, adapté à un capteur sans débit d'air : chute de l'amplitude
respiratoire ≥ 50 % par rapport à la ligne de base pendant 10 à 120 s. L'amplitude
est l'enveloppe crête-à-crête glissante (fenêtre 6 s) ; la ligne de base est le
75e centile de l'enveloppe sur les 3 minutes centrées (robuste aux événements
répétés). Les périodes de mouvement sont exclues. Indice = événements par heure de
sommeil (IDR, « indice de perturbations respiratoires »), à comparer à l'IAH PSG.
Ce n'est PAS un diagnostic : seulement un signal pour consulter.
"""
import numpy as np

FS_R = 10.0


def envelope(w, fs=FS_R, win_s=6.0):
    """Enveloppe crête-à-crête glissante, calculée par blocs de 1 s."""
    n = len(w)
    sec = int(fs)
    ns = n // sec
    mx = w[: ns * sec].reshape(ns, sec).max(axis=1)
    mn = w[: ns * sec].reshape(ns, sec).min(axis=1)
    k = int(win_s)
    env = np.zeros(ns)
    for i in range(ns):
        a, b = max(0, i - k // 2), min(ns, i + k // 2 + 1)
        env[i] = mx[a:b].max() - mn[a:b].min()
    return env  # 1 valeur / seconde


def detect_events(w, moving_s=None, fs=FS_R, drop=0.5, min_s=10, max_s=120, base_win_s=180):
    """→ liste de (début s, fin s, profondeur = 1 − amplitude/ligne de base)."""
    env = envelope(w, fs)
    ns = len(env)
    h = base_win_s // 2
    # ligne de base : 75e centile glissant (calculé toutes les 10 s puis interpolé)
    pts = np.arange(0, ns, 10)
    base_pts = np.array([np.percentile(env[max(0, i - h): i + h + 1], 75) for i in pts])
    base = np.interp(np.arange(ns), pts, base_pts) if len(pts) > 1 else np.full(ns, base_pts[0])
    low = env < (1 - drop) * base
    if moving_s is not None:
        mv = np.zeros(ns, dtype=bool)
        m = np.asarray(moving_s[:ns], dtype=bool)
        mv[: len(m)] = m
        # un mouvement invalide la fenêtre ±10 s
        mvd = np.convolve(mv.astype(float), np.ones(21), mode="same") > 0
        low &= ~mvd
    events = []
    i = 0
    while i < ns:
        if low[i]:
            j = i
            while j < ns and low[j]:
                j += 1
            # l'enveloppe (fenêtre 6 s) raccourcit la chute apparente d'environ 6 s
            dur = (j - i) + 6
            if min_s <= dur <= max_s:
                depth = 1 - np.min(env[i:j]) / max(np.median(base[i:j]), 1e-12)
                events.append((i - 3.0, j + 3.0, float(depth)))
            i = j
        else:
            i += 1
    return events


def rdi(events, sleep_mask_epochs, epoch_s=30):
    """Événements par heure de sommeil (on ne compte que ceux qui débutent en sommeil)."""
    sleep = np.asarray(sleep_mask_epochs, dtype=bool)
    tst_h = sleep.sum() * epoch_s / 3600
    if tst_h <= 0:
        return 0.0, 0
    n = 0
    for s, e, d in events:
        k = int(max(0, s) // epoch_s)
        if k < len(sleep) and sleep[k]:
            n += 1
    return n / tst_h, n
