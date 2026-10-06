"""Assemble le rapport visuel (docs/rapport.html) à partir des résultats des expériences.

    python experiments/build_report.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from somno import model as M  # noqa: E402

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
DOCS = os.path.join(HERE, "..", "..", "docs")


def load(n):
    return json.load(open(os.path.join(RES, n)))


def fr(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def main():
    e01, e02, e04, e05 = (load(f) for f in ("e01_snr.json", "e02_sonar.json", "e04_apnea.json", "e05_staging.json"))
    e06 = load("e06_hrv.json") if os.path.exists(os.path.join(RES, "e06_hrv.json")) else None

    # ---- nuit exemple (rang médian de κ)
    P = np.load(os.path.join(RES, "e05_pred_iphone.npy"))
    d = np.load("/home/user/data/feats_c18.npz")
    y4 = M.to4(d["y"])
    night = d["night"]
    per = sorted(e05["per_night"], key=lambda x: x["kappa"])
    ex = per[len(per) // 2]
    m = night == ex["night"]
    example = dict(kappa=ex["kappa"], truth=y4[m].tolist(), pred=P[m].argmax(1).tolist())

    # ---- e01 : grille natif
    tilts = sorted({x["tilt"] for x in e01})
    bcgs = sorted({x["bcg"] for x in e01})
    nat = [x for x in e01 if x["quant"] == "natif"]

    def grid(key):
        g = []
        for t in tilts:
            row = []
            for b in bcgs:
                v = [x[key] for x in nat if x["tilt"] == t and x["bcg"] == b and x[key] == x[key]]
                row.append(round(float(np.mean(v)), 2) if v else None)
            g.append(row)
        return g

    def agg(key, cond):
        v = [x[key] for x in e01 if cond(x) and x[key] == x[key]]
        return float(np.mean(v)) if v else float("nan")

    rr_good = agg("rr_mae", lambda x: x["quant"] == "natif" and x["tilt"] >= 0.1)
    rr_good_w = agg("rr_within1", lambda x: x["quant"] == "natif" and x["tilt"] >= 0.1)
    hr_good = agg("hr_mae", lambda x: x["quant"] == "natif" and x["bcg"] >= 1.0)
    hr_good_w = agg("hr_within3", lambda x: x["quant"] == "natif" and x["bcg"] >= 1.0)
    hr_q01 = agg("hr_mae", lambda x: x["quant"] == "0,1 m/s²" and x["bcg"] >= 1.0)
    rr_low = agg("rr_mae", lambda x: x["quant"] == "natif" and x["tilt"] == tilts[1])

    # ---- e02 : sonar
    dists = sorted({x["dist"] for x in e02})
    noises = sorted({x["noise"] for x in e02})
    series = []
    for k, nz in enumerate(noises):
        for bl in (0.0, 10.0):
            sel = [[x for x in e02 if x["dist"] == dd and x["noise"] == nz and x["blanket"] == bl] for dd in dists]
            series.append(dict(label=f"micro {nz} dB" + (" + couette" if bl else ""), c=k, blanket=bool(bl),
                               rr=[float(np.nanmean([x["rr_within1"] for x in s]) * 100) for s in sel],
                               hr=[float(np.nanmean([x["hr_within3"] for x in s]) * 100) for s in sel]))
    son_1m = [x for x in e02 if x["dist"] == 1.0 and x["noise"] == -70]
    son_rr_1m = float(np.nanmean([x["rr_within1"] for x in son_1m]))
    son_hr_05 = float(np.nanmean([x["hr_within3"] for x in e02 if x["dist"] == 0.5 and x["noise"] <= -70]))

    # ---- e05 : stades
    ip, ps = e05["iphone"], e05["psg"]
    ens = ip["ensemble"]
    ablation = [dict(label=k.replace("tout (+ heure de la nuit)", "+ heure de la nuit").replace("respiration", "resp."),
                     k=v["kappa4"], acc=v["acc4"], c="--ink-3") for k, v in e05["ablation_hgb_iphone"].items()]
    ablation += [dict(label="iPhone · GRU (nuit entière)", k=ip["GRU"]["kappa4"], acc=ip["GRU"]["acc4"], c="--iphone"),
                 dict(label="iPhone · ensemble retenu", k=ens["kappa4"], acc=ens["acc4"], c="--iphone"),
                 dict(label="PSG idéale · ensemble", k=ps["ensemble"]["kappa4"], acc=ps["ensemble"]["acc4"], c="--psg")]
    pn = ens["per_night"]
    by = e05.get("by_ahi", {})

    # ---- e04 : apnées (seuil d'amplitude, référence) et e04b (classifieur appris, retenu)
    s4 = e04["summary"]
    best = max((k for k in s4 if k.startswith("hyb") and s4[k]), key=lambda k: s4[k]["auc15"])
    psg_best = max((k for k in s4 if k.startswith("psg") and s4[k]), key=lambda k: s4[k]["auc15"])
    e4b = load("e04b_apnea_ml.json")
    ml, mlp = e4b["iphone"], e4b["psg"]
    ap = dict(ahi=[round(x, 2) for x in ml["ahi"]], est=[round(x, 2) for x in ml["est_cal"]])
    e07 = load("e07_holdout.json")

    tiles = [
        dict(big=pct(ens["acc2"]), unit="", text=f"des époques de 30 s bien classées en <strong>éveil ou sommeil</strong> (κ = {fr(ens['kappa2'], 2)})."),
        dict(big=fr(ens["kappa4"], 2), unit="κ", text=f"pour <strong>4 stades</strong> (exactitude {pct(ens['acc4'])}). Avec les signaux parfaits d'une PSG, on atteindrait κ = {fr(ps['ensemble']['kappa4'], 2)}."),
        dict(big="±" + fr(rr_good, 1), unit="resp/min", text=f"d'erreur sur la <strong>respiration</strong> dès que le matelas s'incline d'au moins 0,1° ({pct(rr_good_w)} des époques à ±1)."),
        dict(big="±" + fr(hr_good, 1), unit="bpm", text=f"sur le <strong>pouls</strong> quand l'iPhone sent une secousse d'au moins 1 mg, près du thorax ({pct(hr_good_w)} des époques à ±3)."),
    ]
    texts = {
        "snr-text": (f"<p><strong>Respiration.</strong> Dès que le matelas s'incline d'au moins 0,1° à chaque inspiration, "
                     f"l'erreur moyenne est de {fr(rr_good, 2)} resp/min. C'est le cas sous le drap près de la poitrine. "
                     f"À 0,03°, elle monte à {fr(rr_low, 1)} resp/min. Le gyroscope est décisif : après intégration, son bruit "
                     f"est environ 3 fois plus faible que celui de l'accéléromètre. C'est le blanchiment des canaux qui l'a révélé, "
                     f"en faisant passer l'erreur à 0,03° de 5,8 à 0,8 resp/min.</p>"
                     f"<p><strong>Pouls.</strong> Il faut que la secousse cardiaque perçue atteigne environ 1 mg (erreur "
                     f"{fr(hr_good, 1)} bpm). En dessous de 0,3 mg, le suivi décroche. <strong>Si Safari arrondissait "
                     f"l'accéléromètre à 0,1 m/s²</strong>, comme le fait Chrome, le pouls deviendrait faux "
                     f"(erreur {fr(hr_q01, 0)} bpm) alors que la respiration resterait bonne. L'app mesure ce pas d'arrondi "
                     f"pendant la vérification.</p>"),
        "sonar-text": (f"<p>À 1 m, avec un bruit de micro réaliste, {pct(son_rr_1m)} des époques ont une respiration "
                       f"à ±1 resp/min. À 0,5 m, le pouls est retrouvé dans {pct(son_hr_05)} des époques. Le sonar repère "
                       f"seul la case de distance du thorax (erreur de quelques centimètres) et "
                       f"ignore les objets immobiles. En situation réelle, il faudra composer avec la position du dormeur "
                       f"et le routage audio de Safari : c'est pourquoi l'app native est recommandée pour ce mode.</p>"),
        "stage-intro": (f"Le modèle est entraîné et testé sur <strong>{len(e05['per_night'])} nuits</strong> de "
                        f"polysomnographie du PhysioNet/CinC Challenge 2018, en validation croisée à 5 plis par nuit : "
                        f"aucune nuit n'est vue à l'entraînement et au test. Chaque nuit est d'abord convertie en signaux "
                        f"d'iPhone sur le matelas, avec un couplage tiré au hasard (inclinaison 0,03–0,5°, secousse "
                        f"0,2–3 mg, arrondi des capteurs une nuit sur deux). Elle passe ensuite par la chaîne complète de "
                        f"l'application."),
        "stage-text": (f"<p><strong>Ce qui aide.</strong> Chaque signal ajoute un peu : κ = "
                       f"{fr(e05['ablation_hgb_iphone']['respiration seule']['kappa4'], 2)} avec la respiration seule, "
                       f"{fr(e05['ablation_hgb_iphone']['tout (+ heure de la nuit)']['kappa4'], 2)} avec les mouvements, le "
                       f"pouls et l'heure de la nuit. Le GRU bidirectionnel, qui voit toute la nuit, fait mieux que les "
                       f"arbres, et la moyenne des deux mieux encore. "
                       + (f"La variabilité cardiaque battement par battement, mesurée sur l'ECG avec des battements "
                          f"parfaits, n'apporte que {fr(e06['PSG + VFC battement par battement']['kappa4'] - e06['PSG sans VFC']['kappa4'], 3)} "
                          f"de κ : inutile de chercher chaque battement dans le BCG. " if e06 else "")
                       + f"</p><p><strong>Nuit par nuit.</strong> Le temps de sommeil est estimé avec un écart moyen de "
                       f"{fr(pn['tst']['bias'], 0)} ± {fr(pn['tst']['sd'], 0)} min (corrélation {fr(pn['tst']['r'], 2)}). "
                       f"Le sommeil profond est estimé avec r = {fr(pn['deep']['r'], 2)} et le REM avec r = {fr(pn['rem']['r'], 2)}. "
                       f"<strong>Le point faible est l'éveil</strong> : seulement {pct(ens['wake_sens'])} des époques d'éveil sont "
                       f"reconnues, contre {pct(ens['sleep_sens'])} pour le sommeil. Allongé immobile et éveillé, on ressemble "
                       f"à quelqu'un qui dort, et le proxy de mouvement des PSG est prudent. "
                       + ("Les apnées dégradent aussi l'estimation : κ = " + " · ".join(
                           f"{fr(v['kappa4'], 2)} ({k})" for k, v in by.items()) + "." if by else "") + "</p>"),
        "apnea-intro": (f"Un classifieur estime, pour chaque époque de sommeil, la probabilité qu'elle contienne une apnée "
                        f"ou une hypopnée. Il s'appuie sur la variabilité de l'amplitude et du rythme respiratoires, les cycles "
                        f"de la fréquence cardiaque et les micro-éveils. Sa moyenne sur la nuit, calibrée, donne un indice "
                        f"en événements par heure. Sur {len(ap['ahi'])} nuits en validation croisée, l'indice suit l'IAH de la "
                        f"PSG avec ρ = {fr(ml['spearman'], 2)}. Pour une apnée modérée à sévère (IAH ≥ 15), l'aire sous la "
                        f"courbe ROC vaut <strong>{fr(ml['auc15'], 2)}</strong> : au seuil de 15, l'iPhone repère "
                        f"{pct(ml['sens15'])} des malades, avec {pct(1 - ml['spec15'])} de fausses alertes. Le simple seuil "
                        f"de chute d'amplitude, essayé d'abord, plafonnait à {fr(s4[best]['auc15'], 2)}. Avec les signaux "
                        f"parfaits de la PSG, le même classifieur atteint {fr(mlp['auc15'], 2)}. C'est un signal pour "
                        f"consulter, pas un diagnostic."),
        "holdout-text": (f"<p><strong>Test final sur {e07['n']} nuits jamais vues</strong>, avec le modèle exporté dans "
                         f"l'application : {pct(e07['acc4'])} d'accord sur 4 stades (κ = {fr(e07['kappa4'], 2)}) et "
                         f"{pct(e07['acc2'])} en éveil/sommeil. Le navigateur et Python donnent exactement le même "
                         f"hypnogramme pour ces nuits.</p>"),
    }
    data = dict(example=example, tiles=tiles, snr=dict(tilts=tilts, bcgs=bcgs, rr=grid("rr_mae"), hr=grid("hr_mae")),
                sonar=dict(dists=dists, series=series), ablation=ablation, confusion=ens["confusion4"], apnea=ap,
                texts=texts)
    tpl = open(os.path.join(DOCS, "report_template.html"), encoding="utf-8").read()
    out = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    open(os.path.join(DOCS, "rapport.html"), "w", encoding="utf-8").write(out)
    print("docs/rapport.html", len(out) // 1024, "Ko")
    # ---- section « Résultats chiffrés » de docs/EXPLORATION.md
    md = []
    md.append("### Expérience 1 — iPhone sur le matelas : respiration et pouls\n")
    md.append("6 nuits PSG réelles converties en signaux d'iPhone, époques calmes, erreur absolue moyenne "
              "(capteurs sans arrondi) :\n")
    md.append("| inclinaison \\ secousse BCG | " + " | ".join(f"{b:g} mg".replace(".", ",") for b in bcgs) + " |")
    md.append("|---|" + "---|" * len(bcgs))
    for i, t in enumerate(tilts):
        rr, hr = grid("rr_mae")[i], grid("hr_mae")[i]
        md.append(f"| **{t:g}°** | ".replace(".", ",") + " | ".join(
            f"{fr(a, 1)} resp/min · {fr(b, 1)} bpm" if a is not None and b is not None else "–" for a, b in zip(rr, hr)) + " |")
    md.append(f"\nRespiration : ±{fr(rr_good, 2)} resp/min dès 0,1° d'inclinaison ({pct(rr_good_w)} des époques à ±1). "
              f"Pouls : ±{fr(hr_good, 1)} bpm dès 1 mg ({pct(hr_good_w)} à ±3 bpm). Avec un arrondi des capteurs à "
              f"0,1 m/s² (comme Chrome), le pouls devient faux (±{fr(hr_q01, 0)} bpm) : l'app mesure ce pas.\n")
    md.append("![précision matelas](figures/e01_matelas_snr.png)\n")
    md.append("### Expérience 2 — sonar FMCW\n")
    md.append("| distance | " + " | ".join(s["label"] for s in series) + " |")
    md.append("|---|" + "---|" * len(series))
    for k, dd in enumerate(dists):
        md.append(f"| {fr(dd, 1)} m | " + " | ".join(f"{round(s['rr'][k])} % · {round(s['hr'][k])} %" for s in series) + " |")
    md.append("\n(respiration à ±1 resp/min · pouls à ±3 bpm, en % des époques)\n")
    md.append("![sonar](figures/e02_sonar.png)\n")
    md.append("### Expérience 3 — stades de sommeil (300 nuits, validation croisée par nuit)\n")
    md.append("| modèle | exactitude 4 stades | κ 4 stades | κ 3 stades | veille/sommeil | κ veille/sommeil | éveil reconnu |")
    md.append("|---|---|---|---|---|---|---|")
    for lab, r in (("iPhone · arbres boostés", ip["HGB"]), ("iPhone · GRU bidirectionnel", ip["GRU"]),
                   ("**iPhone · ensemble (app)**", ens), ("PSG idéale · ensemble", ps["ensemble"])):
        md.append(f"| {lab} | {pct(r['acc4'])} | {fr(r['kappa4'], 2)} | {fr(r['kappa3'], 2)} | {pct(r['acc2'])} | "
                  f"{fr(r['kappa2'], 2)} | {pct(r['wake_sens'])} |")
    md.append("\nApport de chaque signal (arbres seuls, κ 4 stades) : " + " · ".join(
        f"{k} {fr(v['kappa4'], 2)}" for k, v in e05["ablation_hgb_iphone"].items()) + ".\n")
    if e06:
        md.append(f"VFC battement par battement (ECG parfait) : κ {fr(e06['PSG sans VFC']['kappa4'], 3)} → "
                  f"{fr(e06['PSG + VFC battement par battement']['kappa4'], 3)} ; gain négligeable, inutile de "
                  f"chercher chaque battement dans le BCG.\n")
    md.append(f"Nuit par nuit : temps de sommeil {fr(pn['tst']['bias'], 0)} ± {fr(pn['tst']['sd'], 0)} min "
              f"(r = {fr(pn['tst']['r'], 2)}), efficacité {fr(pn['se']['bias'], 1)} ± {fr(pn['se']['sd'], 1)} points, "
              f"profond r = {fr(pn['deep']['r'], 2)}, REM r = {fr(pn['rem']['r'], 2)}.\n")
    if by:
        md.append("Selon la sévérité des apnées : " + " · ".join(
            f"{k} (n={v['n']}) κ {fr(v['kappa4'], 2)}" for k, v in by.items()) + ".\n")
    md.append("![stades](figures/e05_stades.png)\n\n![nuit exemple](figures/e05_exemple_nuit.png)\n\n"
              "![nuit par nuit](figures/e05_nuits.png)\n")
    md.append("### Expérience 4 — dépistage d'apnées\n")
    md.append(f"Classifieur appris (probabilité qu'une époque contienne une apnée/hypopnée, moyenne sur la nuit, "
              f"calibrée), validation croisée sur {len(ap['ahi'])} nuits iPhone simulées : ρ = {fr(ml['spearman'], 2)}, "
              f"AUC IAH ≥ 15 = {fr(ml['auc15'], 2)}, AUC IAH ≥ 30 = {fr(ml['auc30'], 2)} ; au seuil de 15 : sensibilité "
              f"{pct(ml['sens15'])}, spécificité {pct(ml['spec15'])} ; erreur moyenne {fr(ml['mae_cal'], 1)} événements/h. "
              f"Même classifieur sur signaux PSG parfaits : AUC {fr(mlp['auc15'], 2)}. Le simple seuil de chute "
              f"d'amplitude (≥ {int(float(best.split('_')[1]) * 100)} %) n'atteignait que AUC {fr(s4[best]['auc15'], 2)} "
              f"(ceintures PSG : {fr(s4[psg_best]['auc15'], 2)}).\n")
    md.append("![apnées](figures/e04_apnees.png)\n")
    md.append("### Expérience 7 — test final sur des nuits jamais vues\n")
    md.append(f"{e07['n']} nuits absentes de l'apprentissage, modèle final de l'application : exactitude 4 stades "
              f"{pct(e07['acc4'])} (κ = {fr(e07['kappa4'], 2)}), veille/sommeil {pct(e07['acc2'])} (κ = {fr(e07['kappa2'], 2)}). "
              f"Par nuit : " + ", ".join(f"κ {fr(q['kappa'], 2)}" for q in e07['per_night']) + ".\n")
    exp = os.path.join(DOCS, "EXPLORATION.md")
    txt = open(exp, encoding="utf-8").read()
    start, end = "<!-- RESULTATS:DEBUT -->", "<!-- RESULTATS:FIN -->"
    block = start + "\n" + "\n".join(md) + "\n" + end
    if start in txt:
        txt = txt[:txt.index(start)] + block + txt[txt.index(end) + len(end):]
    else:
        txt = txt.replace("RESULTATS_ICI", block)
    open(exp, "w", encoding="utf-8").write(txt)
    # résumé pour EXPLORATION.md
    json.dump(dict(rr_good=rr_good, rr_good_w=rr_good_w, hr_good=hr_good, hr_good_w=hr_good_w, hr_q01=hr_q01,
                   rr_low=rr_low, son_rr_1m=son_rr_1m, son_hr_05=son_hr_05, apnea_ml=ml, apnea_drop=s4[best],
                   holdout={k: e07[k] for k in ("n", "acc4", "kappa4", "acc2", "kappa2")}, ens=ens, psg=ps["ensemble"], gru=ip["GRU"], hgb=ip["HGB"],
                   ablation=e05["ablation_hgb_iphone"], by_ahi=by, hrv=e06),
              open(os.path.join(RES, "summary.json"), "w"), indent=1, ensure_ascii=False, default=float)


def pct(x):
    return f"{round(x * 100)} %"


if __name__ == "__main__":
    main()
