"""Classifieur de stades : gradient boosting d'arbres + lissage HMM, export JSON.

Le modèle est entraîné en Python (scikit-learn) puis exporté en JSON compact pour
être évalué tel quel dans l'application iPhone (web/js/model.js, ios/…/Model.swift).
`predict_json` réimplémente l'inférence à partir du JSON : le test vérifie qu'elle
donne exactement les mêmes probabilités que scikit-learn.
"""
import json

import numpy as np

STAGES4 = ["Éveil", "Léger", "Profond", "REM"]
# AASM (0 W, 1 N1, 2 N2, 3 N3, 4 REM) → 4 classes façon Apple Santé
MAP4 = np.array([0, 1, 1, 2, 3])


def to4(y):
    y = np.asarray(y)
    out = np.full(len(y), -1)
    ok = y >= 0
    out[ok] = MAP4[y[ok]]
    return out


def train_hgb(X, y, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=80,
                  l2_regularization=1.0, class_weight="balanced", random_state=0)
    params.update(kw)
    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X, y)
    return clf


def export_hgb(clf, feature_names, extra=None):
    """→ dict JSON : base (n_classes), arbres [[nœuds…] par classe] par itération.
    Nœud feuille : [valeur] ; nœud interne : [feature, seuil, gauche, droite, nan_gauche]."""
    trees = []
    for it in clf._predictors:
        per_class = []
        for tp in it:
            nodes = []
            for nd in tp.nodes:
                if nd["is_leaf"]:
                    nodes.append([round(float(nd["value"]), 6)])
                else:
                    thr = float(nd["num_threshold"])
                    if not np.isfinite(thr):  # JSON n'a pas d'infini : ±1e300 équivaut
                        thr = 1e300 if thr > 0 else -1e300
                    nodes.append([int(nd["feature_idx"]), thr,
                                  int(nd["left"]), int(nd["right"]), int(nd["missing_go_to_left"])])
            per_class.append(nodes)
        trees.append(per_class)
    base = np.asarray(clf._baseline_prediction).ravel().tolist()
    d = dict(kind="hgb", classes=[int(c) for c in clf.classes_], features=list(feature_names),
             base=base, trees=trees)
    if extra:
        d.update(extra)
    return d


def _tree_value(nodes, x):
    i = 0
    while True:
        nd = nodes[i]
        if len(nd) == 1:
            return nd[0]
        v = x[nd[0]]
        if v != v:  # NaN
            i = nd[2] if nd[4] else nd[3]
        else:
            i = nd[2] if v <= nd[1] else nd[3]


def predict_json(model, X):
    """Probabilités (n, n_classes) à partir du modèle JSON (référence pour JS/Swift)."""
    X = np.asarray(X, dtype=np.float64)
    k = len(model["classes"])
    raw = np.tile(np.asarray(model["base"], dtype=np.float64), (len(X), 1))
    for it in model["trees"]:
        for c, nodes in enumerate(it):
            raw[:, c] += [_tree_value(nodes, x) for x in X]
    if k == 2 and raw.shape[1] == 1:
        p1 = 1 / (1 + np.exp(-raw[:, 0]))
        return np.stack([1 - p1, p1], axis=1)
    raw -= raw.max(axis=1, keepdims=True)
    e = np.exp(raw)
    return e / e.sum(axis=1, keepdims=True)


def transition_matrix(y_seqs, k, alpha=1.0):
    T = np.full((k, k), alpha)
    for y in y_seqs:
        y = y[y >= 0]
        for a, b in zip(y[:-1], y[1:]):
            T[a, b] += 1
    return T / T.sum(axis=1, keepdims=True)


def hmm_smooth(P, T, prior, start=None):
    """Lissage avant-arrière : émissions ∝ p(classe|x)/p(classe). → postérieures."""
    n, k = P.shape
    E = P / np.maximum(prior, 1e-9)[None, :]
    E = E / E.sum(axis=1, keepdims=True)
    if start is None:
        start = np.full(k, 1 / k)
    a = np.zeros((n, k))
    a[0] = start * E[0]
    a[0] /= a[0].sum()
    for t in range(1, n):
        a[t] = (a[t - 1] @ T) * E[t]
        a[t] /= a[t].sum()
    b = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        b[t] = T @ (E[t + 1] * b[t + 1])
        b[t] /= b[t].sum()
    post = a * b
    return post / post.sum(axis=1, keepdims=True)


def cohen_kappa(y, p, k):
    m = (y >= 0)
    y, p = y[m], p[m]
    C = np.zeros((k, k))
    np.add.at(C, (y, p), 1)
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(0) * C.sum(1)).sum() / n / n
    return (po - pe) / (1 - pe), C


def save(model, path):
    with open(path, "w") as f:
        json.dump(model, f, separators=(",", ":"), allow_nan=False)
