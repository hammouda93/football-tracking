# Poids de modèles

Les poids ne sont pas inclus dans Git. Placez ici `football-players.pt`, ou changez `YOLO_MODEL_PATH` dans `.env`.

## Classes attendues

Le provider reconnaît les noms suivants :

- `player` / `players`
- `goalkeeper` / `goalie` / `keeper`
- `referee` / `ref`
- `ball` / `football` / `soccer-ball`

Un modèle générique COCO n’est pas suffisant pour un match complet : le ballon occupe parfois seulement quelques pixels et les gardiens/arbitres ont besoin de classes séparées. Entraînez ou adaptez le modèle sur vos angles, résolutions, stades et conditions lumineuses.

## Validation avant production

Mesurez au minimum :

- mAP50-95 par classe, surtout ballon ;
- rappel ballon selon taille/zone de l’image ;
- HOTA, IDF1 et changements d’identité pour les joueurs ;
- erreur de projection terrain en mètres ;
- précision/rappel/F1 par type d’action ;
- calibration de la confiance par tranche de qualité vidéo.

Le dépôt [Roboflow Sports](https://github.com/roboflow/sports) et les tâches [SoccerNet](https://www.soccer-net.org/tasks) fournissent des références et benchmarks. Vérifiez les licences des données et poids avant tout usage commercial.
