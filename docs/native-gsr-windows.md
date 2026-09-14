# Native GSR Windows

`YOLO_PROFILE=native_gsr` est le chemin recommandé lorsque WSL2 n'est pas
disponible. Il s'exécute dans le même `.venv` Windows que Django et conserve
le test continu de 2 min, le live, le ballon, le jeu effectif et les exports.

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
8. OSNet produit une signature d'apparence profonde et réacquiert prudemment
   un joueur après une coupure ; la couleur de peau n'est ni nécessaire ni
   utilisée comme identité.
9. EasyOCR lit les numéros sur plusieurs images. Un vote pondéré au niveau de
   la piste rattache ensuite le numéro à l'effectif importé.
10. Le calibrateur ONNX accepte exactement les 97 points et le schéma du modèle,
    rejette les homographies instables et remet son état à zéro au changement de plan.
11. Une passe multi-échelle récupère périodiquement le petit ballon sans créer
    arbitrairement un ballon loin de tout joueur.

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
NATIVE_GSR_REID_BACKEND=osnet
NATIVE_GSR_REID_MODEL_NAME=osnet_x0_25
NATIVE_GSR_MAX_GAP_SECONDS=3.0
NATIVE_GSR_JERSEY_ENGINE=easyocr
NATIVE_GSR_JERSEY_DEVICE=cpu
NATIVE_GSR_JERSEY_INTERVAL_FRAMES=12
NATIVE_GSR_JERSEY_MAX_CROPS_PER_FRAME=4
NATIVE_GSR_PITCH_MODEL_PATH=
NATIVE_GSR_PITCH_SCHEMA_PATH=
NATIVE_GSR_STRICT_VALIDATION=1
```

Les valeurs centrales du profil sont figées dans les réglages Django afin
qu'un ancien `.env` ne réactive pas silencieusement le prototype. La GTX 1650
doit être utilisée avec `ANALYSIS_DEVICE=0` uniquement si PyTorch confirme
CUDA :

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Avec 4 Go de VRAM, le moteur tente 1280 px puis réduit automatiquement à 960,
768 ou 640 uniquement si CUDA signale une mémoire insuffisante. La résolution
réellement utilisée et chaque réduction apparaissent dans les diagnostics.

## Re-ID et honnêteté du verdict

Sans `NATIVE_GSR_REID_MODEL_PATH`, le moteur calcule un descripteur déterministe
de couleur et texture. Il convient aux liaisons courtes mais ne suffit pas pour
garantir l'identité d'un joueur après une longue disparition. L'interface
affiche alors `histogram` et ajoute un avertissement au verdict.

Le script `install_native_identity_windows.ps1` utilise l'inférence OSNet x0.25
incluse dans le projet, télécharge le checkpoint officiel MSMT17 et vérifie son
SHA-256. Cette couche est compatible avec les poids deep-person-reid mais évite
son extension d'évaluation Cython, inutile ici et fragile à compiler sous Windows.
Il installe aussi
EasyOCR sur CPU afin de conserver les 4 Go de VRAM pour YOLO et OSNet. Le moteur
n'assimile jamais `football-players.pt` à un modèle OCR de maillot.

Sous Windows, l'installation générique d'Ultralytics peut fournir un PyTorch CPU.
Le script le détecte et le remplace par le trio officiel et appairé
`torch 2.11.0`, `torchvision 0.26.0`, `torchaudio 2.11.0` depuis l'index CUDA 12.8,
puis exige que `torch.cuda.is_available()` soit vrai avant de modifier `.env`.

Le raccordement au roster devient automatique uniquement si les CSV des deux
clubs contiennent des numéros réels et uniques. Une lecture ambiguë reste
« inconnue » au lieu d'attribuer un mauvais nom.

## Protocole de validation

1. Conserver le match déjà importé ; aucun nouvel upload n'est requis.
2. Importer les deux effectifs CSV (`name,shirt_number,position`).
3. Recalculer les mi-temps, utiliser « Voir » puis confirmer les quatre limites.
4. Lancer **2. Test · 2 min (1 min/MT)**. Il traite une séquence continue
   de 60 s au centre de la MT1, puis une séquence continue de 60 s au centre
   de la MT2, avec exactement le moteur du match complet.
5. Observer le live final : une seule boîte par objet. Dans la fenêtre Windows,
   `D` affiche ou masque les détections YOLO brutes ; `ESC` arrête le test.
6. Contrôler détection, doublons, ST/CSS, arbitres, gardiens, ballon, stabilité
   des IDs, état OSNet et nombre de pistes ayant un numéro OCR stabilisé.
7. Télécharger la pré-annotation CSV, corriger les boîtes/IDs/équipes/numéros,
   passer `reviewed=YES`, puis la réimporter pour obtenir HOTA@0.50 et IDF1.
8. Lancer le match complet seulement lorsque les bloqueurs stricts sont levés.

Les indicateurs importants sont : joueurs détectés/suivis par image, ballon
visible, IDs tracker vers identités consolidées, fragments reliés, doublons
retirés, pistes par minute et équilibre des deux équipes.

## Ce qui est livré et ce qui ne l'est pas

| Module | État natif Windows |
|---|---|
| Re-ID longue durée | OSNet installé et fusion conservatrice |
| Numéro de maillot | EasyOCR + vote temporel + roster |
| Équipes | apprentissage vidéo, indépendant des couleurs saisies |
| Ballon | passe plein cadre + récupération multi-échelle |
| Terrain | adaptateur ONNX 97 points + RANSAC livré |
| Mesure | CSV humain, IDF1, HOTA@0.50, équipe, maillot, erreur terrain |

Le dépôt ne distribue pas de checkpoint terrain 97 points compatible et licencié.
Il faut fournir **ensemble** le `.onnx` et son schéma sémantique exact. Sans eux,
le test reste exécutable mais la validation stricte signale le terrain comme
bloqueur ; aucune fausse coordonnée n'est fabriquée. HOTA@0.50 est un contrôle
local à un seuil, pas le HOTA multi-seuil ni le GS-HOTA officiel.

Le runner officiel TrackLab + `sn-gamestate` reste disponible comme référence
externe GPL-3.0 lorsque Linux/WSL fonctionne.

## Licences et références

- [TrackLab](https://github.com/TrackingLaboratory/tracklab) — MIT.
- [deep-person-reid / OSNet](https://github.com/KaiyangZhou/deep-person-reid) —
  architecture OSNet adaptée sous licence MIT pour l'inférence locale.
- [SoccerNet sn-gamestate](https://github.com/SoccerNet/sn-gamestate) — GPL-3.0.
- [SoccernetGSR Winner 2025](https://github.com/yinmayoo185/SoccernetGSR) —
  étude technique uniquement tant qu'aucune licence n'est publiée à la racine.
