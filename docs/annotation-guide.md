# Guide d’annotation et de validation

## Règle générale

Une action doit avoir un instant reproductible, un acteur lorsque visible, une équipe, un résultat et une preuve vidéo. Si l’acteur est hors champ ou le ballon invisible, conserver `pending` ou `unknown` plutôt que deviner.

| Action | Instant de référence | Succès | Échec |
|---|---|---|---|
| Passe | première touche du destinataire | même équipe contrôle | adversaire/hors jeu contrôle |
| Conduite | fin du déplacement contrôlé | joueur conserve | perte pendant la conduite |
| Dribble | sortie du duel | attaquant conserve | défenseur récupère |
| Duel | premier contact disputé | acteur gagne la possession | acteur perd la possession |
| Tir | contact avec le ballon | résultat précisé par qualifier | résultat précisé par qualifier |
| Récupération | contrôle stable après ballon adverse/libre | contrôle établi | — |
| Perte | fin du dernier contrôle | — | adversaire contrôle |
| Sortie | ballon franchit entièrement la ligne | neutre | neutre |

## Niveaux de confiance conseillés

- `≥ 0.92` : visible, identité stable et transition non ambiguë ; auto-acceptation possible.
- `0.70–0.92` : action probable mais validation rapide requise.
- `< 0.70` : candidat de rappel ; ne pas intégrer dans un rapport officiel sans validation.

## Dataset local

Exporter périodiquement les événements corrigés avec leurs clips. Séparer les matches par compétition, stade, caméra et saison avant de créer les ensembles entraînement/validation/test afin d’éviter qu’une même apparence de match fuite entre les ensembles.
