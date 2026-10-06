# Somno — application iOS native (optionnelle)

L'application web suffit pour commencer. L'app native apporte ce que Safari ne permet pas :

| | App web (Safari) | App native |
|---|---|---|
| Écran pendant la nuit | allumé (noir), Safari au premier plan | **éteint** (mode d'arrière-plan audio) |
| Capteurs de mouvement | ~60 Hz, précision fixée par Safari | **100 Hz** CoreMotion, pleine résolution, moyenne → 50 Hz |
| Sonar ultrasonore | expérimental (dépend du routage audio de Safari) | fiable : `AVAudioSession` mode `.measurement`, sans annulation d'écho |
| Caméra de profondeur TrueDepth | impossible | **mode expérimental** (respiration sans contact, dans le noir) |
| Apple Santé | impossible | **écrit les phases de sommeil** (Éveillé / Principal / Profond / Paradoxal) |

Pour ne pas maintenir trois implémentations, l'app native :
- **enregistre** les capteurs en Swift (CoreMotion, AVAudioEngine, AVCaptureDepthDataOutput) ;
- exécute le **même code JavaScript** que l'app web pour le son et le sonar
  (`web/core/somno-core.js`, généré par `node web/tools/build-core.mjs`, dans JavaScriptCore) ;
- écrit un fichier **`.somno`** (même format que l'export web) ;
- l'**analyse** et le **rapport** sont ceux de l'app web, affichés dans un `WKWebView`
  (`ReportView.swift`), qui renvoie l'hypnogramme à Swift pour Apple Santé.

> ⚠️ Ce code n'a pas pu être compilé dans l'environnement où il a été écrit (Linux, pas
> d'Xcode). La logique de traitement partagée est testée (Python ⇄ JS ⇄ script JavaScriptCore),
> mais attendez-vous à quelques retouches de compilation Swift.

## Installer sur votre iPhone 14 Pro (compte Apple gratuit)

1. Sur un Mac avec Xcode 15+ : `brew install xcodegen`
2. `cd sommeil && node web/tools/build-core.mjs` (génère `web/core/somno-core.js`)
3. `cd ios && xcodegen && open Somno.xcodeproj`
4. Dans Xcode › cible Somno › *Signing & Capabilities* : choisissez votre équipe (votre identifiant
   Apple). Si la capacité **HealthKit** est refusée avec un compte gratuit, supprimez-la
   (et `Somno.entitlements`) : seule l'écriture dans Santé sera désactivée.
5. Branchez l'iPhone, sélectionnez-le, ▶︎. Sur l'iPhone : Réglages › Général › VPN et gestion
   de l'appareil › faire confiance au développeur. Activez le **mode développeur** si demandé.
   (Avec un compte gratuit, l'app doit être réinstallée tous les 7 jours.)

## Utilisation

- **Sur le matelas** : iPhone à plat sous le drap-housse ou sous l'oreiller, au niveau de la
  poitrine. Écran verrouillable. Branchez le chargeur.
- **Table de nuit · sonar** : bas de l'iPhone (haut-parleur + micro) tourné vers vous, < 1 m.
  Émet des ultrasons 18–20,5 kHz (inaudibles pour la plupart des adultes ; chiens et chats
  les entendent).
- **Support · TrueDepth** : écran face au lit, 0,5–1 m, sur un support. L'écran reste allumé (noir).

Le matin, touchez la nuit → analyse et rapport. Le bouton de partage exporte le `.somno`
(analyse Python : `python -m somno.cli nuit.somno --out nuit.png`). Les extraits sonores
(`.wav`) sont dans l'app Fichiers › Somno › nuits.

## Fichiers

| Fichier | Rôle |
|---|---|
| `NightRecorder.swift` | orchestre une nuit, écrit le `.somno` |
| `MotionCapture.swift` | CoreMotion 100 Hz → grille 50 Hz |
| `AudioCapture.swift` | micro + chirps sonar, traitement par `somno-core.js` dans JavaScriptCore |
| `DepthCapture.swift` | profondeur TrueDepth, grille 4×4, 15 i/s |
| `SomnoWriter.swift` | format `.somno` |
| `ReportView.swift` | app web embarquée (schéma `somno://`), retour du rapport |
| `HealthExport.swift` | écriture des phases dans Apple Santé |
