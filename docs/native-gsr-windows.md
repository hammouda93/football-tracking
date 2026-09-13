# Native GSR Windows

`YOLO_PROFILE=native_gsr` est le chemin recommandé lorsque WSL2 n'est pas
disponible. Il s'exécute dans le même `.venv` Windows que Django et conserve les
tests 40 s / 2 min, le live, le ballon, le jeu effectif et les exports.

## Architecture réellement exécutée

1. Le modèle football détecte ballon, gardiens, joueurs et arbitres à 1280 px.
2. BoT-SORT associe les personnes et compense les mouvements de caméra.
3. Un second étage supprime les doubles boîtes qui décrivent la même personne.
4. L'apparence est agrégée sur toute la piste, jamais décidée sur une seule image.
5. Les fragments séparés sont réunis seulement si apparence, rôle, temps et
   déplacement sont tous compatibles. Une ambiguïté conserve deux IDs distincts.
6. Les deux maillots sont appris par K-means sur les torses de plusieurs
   tracklets. Les couleurs saisies à l'import ne participent pas au calcul.
7. La correspondance groupe A/B vers ST/CSS reste inversable dans l'interface.
8. La position image utilise le bas-centre de la boîte. La position terrain en
   mètres n'est produite que lorsqu'une homographie valable est disponible.

Cette structure reprend les principes publiés de TrackLab/`sn-gamestate`
(composants séparés, état par tracklet, apparence, équipe, projection terrain)
et les idées décrites par SoccernetGSR Winner 2025 (détections faibles de
récupération, compensation caméra, Re-ID et raffinement après tracking). Le code
de ce dépôt est une implémentation originale. Aucun fichier du Winner 2025 n'est
copié, car son dépôt ne publie pas de licence de réutilisation.

## Configuration recommandée

```env
ANALYSIS_BACKEND=yolo
ANALYSIS_ATHLETE_ENGINE=legacy
ANALYSIS_TRACKING_FPS=12.5
ANALYSIS_MIN_YOLO_TRACKING_FPS=12.5
ANALYSIS_DEVICE=0
ANALYSIS_LIVE_WINDOW=1

YOLO_PROFILE=native_gsr
YOLO_MODEL_PATH=models/football-players.pt
YOLO_CONFIDENCE=0.18
YOLO_BALL_CONFIDENCE=0.12
YOLO_IMAGE_SIZE=1280
YOLO_TRACKER=botsort
YOLO_TRACK_LOW_CONFIDENCE=0.05
YOLO_NEW_TRACK_CONFIDENCE=0.20
YOLO_TRACK_MATCH_THRESHOLD=0.80
YOLO_TRACK_BUFFER_SECONDS=4.8

YOLO_PLAYER_CLASS_IDS=2
YOLO_GOALKEEPER_CLASS_IDS=1
YOLO_REFEREE_CLASS_IDS=3
YOLO_BALL_CLASS_IDS=0

NATIVE_GSR_REID_MODEL_PATH=
NATIVE_GSR_MAX_GAP_SECONDS=3.0
```

Les valeurs centrales du profil sont figées dans les réglages Django afin
qu'un ancien `.env` ne réactive pas silencieusement le prototype. La GTX 1650
doit être utilisée avec `ANALYSIS_DEVICE=0` uniquement si PyTorch confirme
CUDA :

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Re-ID et honnêteté du verdict

Sans `NATIVE_GSR_REID_MODEL_PATH`, le moteur calcule un descripteur déterministe
de couleur et texture. Il convient aux liaisons courtes mais ne suffit pas pour
garantir l'identité d'un joueur après une longue disparition. L'interface
affiche alors `histogram` et ajoute un avertissement au verdict.

Un checkpoint compatible d'embeddings peut être indiqué dans
`NATIVE_GSR_REID_MODEL_PATH`. Le moteur n'assimile jamais
`football-players.pt` à un modèle OCR de maillot : le premier détecte des objets,
le second problème demande des poids Re-ID/OCR spécialisés et une évaluation
séparée.

## Protocole de validation

1. Conserver le match déjà importé ; aucun nouvel upload n'est requis.
2. Vérifier et confirmer les limites des deux mi-temps.
3. Lancer le test court de 40 s et observer le live.
4. Contrôler la détection, les doublons, ST/CSS, arbitres et ballon.
5. Lancer le test continu de 2 min seulement si les huit extraits sont crédibles.
6. Autoriser le match complet seulement si le test 2 min est validé, ou après
   validation humaine explicite d'un avertissement compris.

Les indicateurs importants sont : joueurs détectés/suivis par image, ballon
visible, IDs tracker vers identités consolidées, fragments reliés, doublons
retirés, pistes par minute et équilibre des deux équipes.

## Ce qui reste nécessaire pour atteindre la référence GSR

- poids Re-ID football évalués sur cette caméra ;
- OCR de numéro avec vote au niveau du tracklet ;
- réseau de points-clés du terrain et calibration automatique par plan ;
- ball tracker spécialisé à haute fréquence ;
- vérité terrain annotée pour mesurer HOTA, IDF1, mAP et erreur métrique.

Ces modules sont des étapes mesurables, pas des résultats simulés. Le runner
officiel TrackLab + `sn-gamestate` reste disponible comme référence externe
GPL-3.0 lorsque WSL/Linux fonctionne.

## Licences et références

- [TrackLab](https://github.com/TrackingLaboratory/tracklab) — MIT.
- [SoccerNet sn-gamestate](https://github.com/SoccerNet/sn-gamestate) — GPL-3.0.
- [SoccernetGSR Winner 2025](https://github.com/yinmayoo185/SoccernetGSR) —
  étude technique uniquement tant qu'aucune licence n'est publiée à la racine.

