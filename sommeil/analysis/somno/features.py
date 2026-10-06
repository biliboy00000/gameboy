"""Assemblage des caractéristiques par époque + normalisation par nuit + contexte.

Le classifieur ne voit jamais de valeurs absolues seules : chaque grandeur est aussi
exprimée relativement à la nuit de la personne (médiane/MAD), ce qui rend le modèle
insensible au fait que tel dormeur a une FC de repos à 48 ou à 70 bpm, ou que
l'iPhone est plus ou moins loin du thorax (amplitude respiratoire).
Le contexte temporel (moyennes/écarts-types glissants centrés) imite ce que fait un
scoreur humain : on ne décide pas d'un stade sur 30 s isolées.
"""
import numpy as np

from . import dsp

RESP = ["rr", "rr_spec", "ibi_cv", "amp_cv", "reg", "spec_conc", "amp", "n_breaths"]
HEART = ["hr", "hr_std", "hr_q"]
MOV = ["mov_frac", "mov_int", "t_since_mov", "t_to_mov"]
BASE = RESP + HEART + MOV

# grandeurs normalisées par nuit (z robuste)
NORM = ["rr", "rr_spec", "ibi_cv", "amp_cv", "reg", "spec_conc", "hr", "hr_std"]
# grandeurs dont on ajoute le contexte (moyenne et écart-type glissants)
CTX = ["rr_z", "ibi_cv", "amp_cv", "reg", "spec_conc", "amp_log", "hr_z", "hr_std",
       "mov_frac"]
CTX_WIN = [5, 11, 21]  # époques (2,5 / 5,5 / 10,5 min)


def rolling(x, w, fn):
    """Statistique glissante centrée ignorant les NaN (vectorisé par cumul)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    h = w // 2
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    c1 = np.concatenate([[0], np.cumsum(v)])
    c2 = np.concatenate([[0], np.cumsum(v * v)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    cnt = cn[hi] - cn[lo]
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        if fn == "mean":
            return mean
        var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mean * mean, np.nan)
        return np.sqrt(np.maximum(var, 0))


def assemble(resp_F, hr_F, mov_F):
    """Concatène et dérive les caractéristiques. → (matrice (n_ep, k), noms)."""
    n = min(len(resp_F), len(hr_F), len(mov_F))
    X = np.concatenate([resp_F[:n], hr_F[:n], mov_F[:n]], axis=1)
    cols = {name: X[:, i] for i, name in enumerate(BASE)}
    out = dict(cols)
    for name in NORM:
        v = cols[name]
        med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        out[name + "_z"] = (v - med) / dsp.robust_scale(v)
    amp = cols["amp"]
    med_amp = np.nanmedian(amp[amp > 0]) if (amp > 0).any() else 1.0
    out["amp_log"] = np.log(np.maximum(amp, 1e-12) / med_amp)
    # position dans la nuit (les cycles : N3 tôt, REM tard)
    t_h = (np.arange(n) + 0.5) * 30 / 3600
    out["t_hours"] = t_h
    out["t_frac"] = t_h / max(t_h[-1], 1e-6)
    for name in CTX:
        for w in CTX_WIN:
            out[f"{name}_m{w}"] = rolling(out[name], w, "mean")
            out[f"{name}_s{w}"] = rolling(out[name], w, "std")
    names = list(out.keys())
    return np.stack([out[k] for k in names], axis=1), names
