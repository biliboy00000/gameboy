"""Simulateur physique d'un iPhone posé sur le matelas.

Sert à deux choses :
  1. tester les algorithmes quand on connaît la vérité (respiration, FC) ;
  2. « hybride PSG » : convertir une vraie nuit de polysomnographie (ceintures
     respiratoires, battements ECG, mouvements) en signaux que l'iPhone aurait
     enregistrés, pour entraîner/valider le classifieur de stades sur la chaîne
     complète de traitement du téléphone.

Modèle (ordres de grandeur issus de la littérature BCG / SCG sur matelas) :
  - respiration → inclinaison du matelas θ(t), 0,03–0,5° selon la distance au thorax
    → projection de la gravité sur x/y (≈ g·θ) + vitesse angulaire dθ/dt au gyroscope
    + petit déplacement vertical (1–3 mm) ;
  - battement → salve BCG (ondes I-J-K, 4–10 Hz) ≈ 0,2–3 mg et 0,02–0,3 °/s,
    modulée par la respiration, gain qui change à chaque changement de position ;
  - mouvements → rafales larges bande, nouvelle orientation du téléphone ;
  - capteurs : bruit blanc (accéléromètre ≈ 150 µg/√Hz, gyroscope ≈ 0,005 °/s/√Hz),
    quantification optionnelle (Safari arrondit-il ? → on teste plusieurs pas).
"""
import numpy as np

G = 9.80665
DEG = np.pi / 180


def rot_small(theta_x, theta_y):
    """Gravité vue par le téléphone pour de petites inclinaisons (rad)."""
    gx = G * np.sin(theta_y)
    gy = -G * np.sin(theta_x) * np.cos(theta_y)
    gz = G * np.cos(theta_x) * np.cos(theta_y)
    return np.stack([gx, gy, gz], axis=1)


def bcg_template(fs, rng):
    """Forme d'une salve BCG : sinusoïde amortie (H-I-J-K-L), 0,5 s."""
    t = np.arange(int(0.6 * fs)) / fs
    f0 = rng.uniform(5.0, 8.0)
    tau = rng.uniform(0.06, 0.12)
    env = (t / tau) * np.exp(1 - t / tau)
    return env * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))


def synth_phone(resp, beats_s, moving_s, fs=50.0, seed=0, duration_s=None,
                resp_tilt_deg=None, bcg_mg=None, acc_noise_ug=150.0,
                gyro_noise_dps=0.005, quant_acc=0.0, quant_gyro=0.0,
                partner_rate_h=0.0, fs_resp=None):
    """Synthétise accéléromètre (n,3, m/s²) et gyroscope (n,3, rad/s) à fs Hz.

    resp      : effort respiratoire (n_r,) à fs_resp Hz, unité arbitraire
    beats_s   : instants des battements (s) — ex. pics R de l'ECG + délai
    moving_s  : booléen par seconde, la personne bouge
    """
    rng = np.random.default_rng(seed)
    if duration_s is None:
        duration_s = len(moving_s)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # --- respiration → inclinaison (normalisée par l'amplitude robuste de la nuit)
    r = np.interp(t, np.arange(len(resp)) / fs_resp, resp)
    r = r - np.median(r)
    scale = np.percentile(np.abs(r), 90) or 1.0
    r = np.clip(r / scale, -4, 4)
    if resp_tilt_deg is None:
        resp_tilt_deg = float(np.exp(rng.uniform(np.log(0.03), np.log(0.5))))
    if bcg_mg is None:
        bcg_mg = float(np.exp(rng.uniform(np.log(0.2), np.log(3.0))))

    # --- segments de position : chaque épisode de mouvement change la configuration
    mv = np.asarray(moving_s, dtype=bool)[: int(duration_s)]
    seg_id = np.cumsum(np.concatenate([[0], (mv[1:] & ~mv[:-1]).astype(int)]))
    n_seg = int(seg_id[-1]) + 1 if len(seg_id) else 1
    base_tilt = rng.normal(0, 3 * DEG, size=(n_seg, 2))           # téléphone pas tout à fait à plat
    resp_dir = rng.normal(0, 1, size=(n_seg, 2))
    resp_dir /= np.linalg.norm(resp_dir, axis=1, keepdims=True)
    resp_gain = np.exp(rng.normal(0, 0.5, size=n_seg))           # couplage selon la position
    bcg_gain = np.exp(rng.normal(0, 0.6, size=n_seg))
    bcg_dir = rng.normal(0, 1, size=(n_seg, 6))
    bcg_dir /= np.linalg.norm(bcg_dir, axis=1, keepdims=True)
    seg_t = np.repeat(seg_id, int(fs))[:n]
    if len(seg_t) < n:
        seg_t = np.concatenate([seg_t, np.full(n - len(seg_t), seg_t[-1] if len(seg_t) else 0)])

    amp = resp_tilt_deg * DEG * resp_gain[seg_t]
    th = base_tilt[seg_t] + (r * amp)[:, None] * resp_dir[seg_t]
    acc = rot_small(th[:, 0], th[:, 1])
    # déplacement vertical du matelas (≈ 2 mm crête) → accélération
    z = 0.002 * r * resp_gain[seg_t]
    acc[:, 2] += np.gradient(np.gradient(z, 1 / fs), 1 / fs)
    gyro = np.zeros((n, 3))
    gyro[:, 0] = np.gradient(th[:, 0], 1 / fs)
    gyro[:, 1] = np.gradient(th[:, 1], 1 / fs)

    # --- BCG
    tpl = bcg_template(fs, rng)
    pulses = np.zeros(n)
    bi = np.round(np.asarray(beats_s) * fs).astype(int)
    bi = bi[(bi >= 0) & (bi < n)]
    resp_mod = 1 + 0.2 * np.clip(r, -2, 2)
    pulses[bi] = 1.0 * resp_mod[bi] * np.exp(rng.normal(0, 0.15, size=len(bi)))
    wave = np.convolve(pulses, tpl, mode="full")[:n]
    g6 = (bcg_gain[seg_t] * bcg_mg * 1e-3 * G)[:, None] * bcg_dir[seg_t]
    acc += wave[:, None] * g6[:, :3]
    gyro += wave[:, None] * g6[:, 3:] * (0.1 * DEG / (1e-3 * G))  # ≈ 0,1 °/s par mg

    # --- mouvements de la personne (et du/de la partenaire)
    mvt = np.repeat(mv, int(fs))[:n].astype(float)
    if partner_rate_h > 0:
        k = rng.poisson(partner_rate_h * duration_s / 3600)
        for s in rng.uniform(0, duration_s - 10, size=k):
            a, b = int(s * fs), int((s + rng.uniform(1, 6)) * fs)
            mvt[a:b] = np.maximum(mvt[a:b], 0.15)  # atténué : transmis par le matelas
    if mvt.any():
        burst_a = rng.normal(0, 1, size=(n, 3)) * 0.8
        burst_g = rng.normal(0, 1, size=(n, 3)) * 20 * DEG
        from .dsp import filtfilt
        burst_a = filtfilt(burst_a, fs, hi=6.0, order=2)
        burst_g = filtfilt(burst_g, fs, hi=6.0, order=2)
        acc += burst_a * mvt[:, None]
        gyro += burst_g * mvt[:, None]

    # --- bruit et quantification des capteurs
    bw = np.sqrt(fs / 2)
    acc += rng.normal(0, acc_noise_ug * 1e-6 * G * bw, size=acc.shape)
    gyro += rng.normal(0, gyro_noise_dps * DEG * bw, size=gyro.shape)
    if quant_acc > 0:
        acc = np.round(acc / quant_acc) * quant_acc
    if quant_gyro > 0:
        gyro = np.round(gyro / quant_gyro) * quant_gyro
    info = dict(resp_tilt_deg=resp_tilt_deg, bcg_mg=bcg_mg, n_segments=n_seg)
    return acc, gyro, info


def synthetic_night(hours=7.5, seed=0, fs_resp=10.0):
    """Nuit entièrement synthétique avec vérité terrain (pour tests unitaires) :
    respiration à fréquence/variabilité dépendant du stade, FC idem, mouvements
    pendant l'éveil. → dict avec stades (30 s), resp, battements, mouvements."""
    rng = np.random.default_rng(seed)
    n_ep = int(hours * 120)
    # hypnogramme simple : cycles de 90 min, N3 en début de nuit, REM en fin
    stages = np.zeros(n_ep, dtype=int)
    e = 20  # 10 min d'endormissement
    while e < n_ep:
        cyc = min(n_ep - e, int(rng.uniform(80, 110) * 2))
        frac = e / n_ep
        n3 = int(cyc * max(0.0, 0.35 - 0.4 * frac))
        rem = int(cyc * (0.1 + 0.25 * frac))
        n1 = int(cyc * 0.06)
        n2 = cyc - n3 - rem - n1
        seq = [1] * n1 + [2] * (n2 // 2) + [3] * n3 + [2] * (n2 - n2 // 2) + [4] * rem
        stages[e:e + len(seq)] = seq[: n_ep - e]
        e += cyc
        if rng.random() < 0.5 and e < n_ep:
            w = int(rng.uniform(1, 6))
            stages[e:e + w] = 0
            e += w
    stages[-6:] = 0
    # respiration : fréquence et irrégularité selon le stade
    rate = {0: 15, 1: 14.5, 2: 14, 3: 13, 4: 16}
    irr = {0: 0.25, 1: 0.15, 2: 0.08, 3: 0.04, 4: 0.2}
    hr_m = {0: 68, 1: 62, 2: 58, 3: 55, 4: 63}
    hr_v = {0: 4, 1: 2.5, 2: 1.5, 3: 1.0, 4: 3.5}
    resp, beats, mov = [], [], []
    phase, t_beat = 0.0, 0.5
    dt = 1 / fs_resp
    rr_now = 14.0
    hr_now = 60.0
    for ep in range(n_ep):
        s = stages[ep]
        for _ in range(int(30 * fs_resp)):
            rr_now += 0.02 * (rate[s] - rr_now) + rng.normal(0, irr[s]) * 0.3
            phase += 2 * np.pi * max(rr_now, 4) / 60 * dt
            ampl = 1 + (rng.normal(0, irr[s]) if rng.random() < 0.05 else 0)
            resp.append(ampl * np.sin(phase))
        t_end = (ep + 1) * 30.0
        while t_beat < t_end:
            hr_now += 0.05 * (hr_m[s] - hr_now) + rng.normal(0, hr_v[s]) * 0.3
            beats.append(t_beat)
            t_beat += 60 / np.clip(hr_now, 40, 120)
        for _ in range(30):
            p = {0: 0.3, 1: 0.03, 2: 0.008, 3: 0.002, 4: 0.01}[s]
            mov.append(rng.random() < p)
    mov = np.array(mov)
    # les mouvements durent quelques secondes
    mov = np.convolve(mov.astype(float), np.ones(4), mode="same") > 0
    return dict(stages=stages, resp=np.array(resp), fs_resp=fs_resp,
                beats=np.array(beats), moving=mov)
