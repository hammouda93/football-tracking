# Sources et idées retenues

Cette note décrit les références qui ont influencé l’architecture, sans copier leurs implémentations ni promettre leurs résultats.

## Produit de référence : Impact Soccer

[Impact Soccer](https://mpact.ai/) présente un flux simple : importer un match complet, obtenir des statistiques équipe/joueur et générer des clips. Sa [FAQ](https://mpact.ai/faqs) insiste aussi sur trois réalités reprises dans ce projet : le fichier brut est préférable à un lien recompressé, la qualité/position de caméra détermine la qualité des résultats, et l’attribution individuelle utilise le roster et la lisibilité des numéros.

Idées intégrées :

- upload d’un match complet plutôt que clips pré-découpés ;
- statistiques d’équipe d’abord, attribution joueurs ensuite ;
- clips horodatés et résultats téléchargeables ;
- diagnostic de qualité et avertissements explicites ;
- architecture locale pour garder le contrôle de la vidéo.

## Démonstrations vidéo utiles

- [Football AI Tutorial: From Basics to Advanced Stats with Python](https://www.youtube.com/watch?v=aBVGKoNZQUw) : détection, tracking, équipes, possession et vitesse dans une chaîne lisible.
- [Build an AI/ML Football Analysis system with YOLO, OpenCV…](https://www.youtube.com/watch?v=neBZ6huolkg) : séparation détection → tracking → compensation caméra → métriques.
- [Football Players Tracking — YOLOv5 + ByteTrack](https://www.youtube.com/watch?v=QCG8QMhga9k) : identités temporaires par multi-object tracking.
- [Football AI Community Q&A](https://www.youtube.com/watch?v=Xwou5qO--vY) : limites pratiques, datasets et dépôt Roboflow Sports.
- [Real-Time Football Player and Ball Detection](https://www.youtube.com/watch?v=z6jmZltVeGA) : importance d’un modèle spécialisé ballon/joueurs.

Ces vidéos sont de bonnes démonstrations techniques, mais une courte séquence annotée n’est pas une preuve de précision sur 90 minutes. Le projet ajoute donc les périodes, les coupures, la double horloge, la persistance, les runs et la validation humaine.

## Références ouvertes et publications

- [Roboflow Sports](https://github.com/roboflow/sports) : détection sportive, points-clés du terrain, classification d’équipe et exemples de projection.
- [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate) : benchmark de localisation/identification depuis une caméra mobile et reconstruction en vue terrain.
- [Article CVPRW 2024 SoccerNet-GSR](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf) : formalisation du suivi, de l’identité et de la calibration comme un même problème.
- [SoccerNet Action Spotting](https://github.com/SoccerNet/sn-spotting) : matches complets non découpés et événements temporels.
- [Ultralytics Tracking](https://docs.ultralytics.com/modes/track/) : trackers disponibles et intégration détection/tracking.

## Décision importante

Le projet n’utilise pas un grand modèle de langage pour « regarder » chaque frame et inventer des statistiques. La vision produit des signaux structurés ; les règles et modèles temporels produisent des candidats ; l’analyste contrôle les cas ambigus. Un assistant IA pourra ensuite expliquer les données validées, jamais remplacer la preuve vidéo.
