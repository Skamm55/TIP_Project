# Choix du modèle final (tâche 4.6)

Objet : valider le modèle soumis au leaderboard (tâches 5.1 à 5.4) à partir des analyses 4.1 à 4.5.

Critère de décision : le score de frugalité du challenge, Frug = F1 − 0,3·R / (R + 1) (R en millions de
paramètres), qui classe le leaderboard. Le F1 macro sert de critère secondaire.

Données : 6 runs, évalués sur la même validation (2 660 images, split stratifié, graine 42).

## Étape 1 : CNN du TD ou transfer learning ? → Transfer learning

- La baseline du TD plafonne à un F1 de 0,335 (frugalité 0,178), contre 0,86 à 0,93 pour tous les modèles
  pré-entraînés.
- Diagnostic (4.2) : biais élevé. Train et validation restent tous deux vers 37 % : une couche de 8 filtres
  ne suffit pas pour 11 classes d'aliments.
- Le cours prescrit dans ce cas un réseau plus gros ou une autre architecture (complementary slides, p.28),
  ce qu'apporte précisément un réseau pré-entraîné.
- La 4.5 le confirme : les premières couches détectent des motifs génériques (contours, couleurs, textures),
  réutilisables tels quels depuis ImageNet (CVML lectures, p.83).

## Étape 2 : gelé ou fine-tuning ? → Fine-tuning

- Pour un même réseau, le fine-tuning gagne 5 points de F1 (MobileNetV2 : 0,874 → 0,925 ;
  ResNet-18 : 0,859 → 0,913).
- À frugalité, le gain est entier : geler ne réduit pas R, puisque le challenge compte tous les paramètres.
  Les deux modes ont la même pénalité et seul le F1 les départage.
- Diagnostic (4.2) : les modèles gelés plafonnent dès les époques 3 à 8, un biais modéré que seul le dégel
  permet de lever.
- Contrepartie : un début de variance en fine-tuning (écart train / validation de 6 à 7 points),
  traité à l'étape 4.

## Étape 3 : MobileNetV2 ou ResNet-18 ? → MobileNetV2

| | MobileNetV2 fine-tuning | ResNet-18 fine-tuning |
|---|---|---|
| F1 de validation | **0,925** | 0,913 |
| Paramètres | **2,24 M** | 11,18 M |
| Pénalité | **0,207** | 0,275 |
| Frugalité | **0,718** | 0,638 |

- MobileNetV2 gagne sur les deux critères à la fois : il n'y a pas de compromis à arbitrer.
- Pour l'égaler, ResNet-18 devrait atteindre un F1 d'au moins 0,993, ce qui est irréaliste (4.1).
- Même MobileNetV2 gelé (0,667) bat ResNet-18 en fine-tuning (0,638) : le choix du réseau pèse plus que
  la stratégie d'entraînement.
- Seul point en faveur de ResNet-18 : une époque plus rapide sur GPU (27 s contre 42 s), critère qui ne
  compte pas dans le challenge.

## Étape 4 : avec ou sans augmentation ? → Avec augmentation

- Avec augmentation : F1 de 0,925 contre 0,911, frugalité de 0,718 contre 0,703.
- Sans augmentation, la variance augmente nettement (4.4) : accuracy train de 99,6 % dès l'époque 9,
  écart train / validation de 6,2 à 10,0 points, minimum de la loss de validation 5 époques plus tôt.
- L'écart de F1 (0,014) est modeste mais va dans le même sens que les courbes : la décision repose sur
  cette convergence d'indices, pas sur le seul F1.

## Étape 5 : quel point de l'entraînement garder ? → Le meilleur F1 de validation, époque 15

- `train.py` garde automatiquement le modèle au meilleur F1 macro de validation, la métrique du serveur.
- Le meilleur F1 tombe à la dernière époque, mais les courbes montrent un plateau (+0,005 sur les
  4 dernières époques) : relancer plus longtemps accentuerait surtout le surapprentissage (4.2).
- La loss de validation atteint son minimum à l'époque 11, mais le critère du challenge est le F1.

## Étape 6 : les erreurs remettent-elles le choix en cause ? → Non

- 225 erreurs sur 2 660 (8,5 %), concentrées sur dessert, bread, fried et meat (4.3), entre classes
  visuellement proches, déjà repérées dans `DATA.md`.
- Une bonne part vient du dataset lui-même : plusieurs aliments par assiette pour une seule étiquette, et
  des catégories qui se recouvrent (« fried » est un mode de cuisson). Aucun autre modèle testé ne corrige
  ce type d'erreur.
- Le déséquilibre n'explique pas les erreurs : `rice`, la plus petite classe, est la mieux reconnue (F1 0,987).

## Étape 7 : le choix est-il robuste ? → Oui

- Unité de R (pas encore confirmée par le prof) : avec R en nombre brut, la pénalité vaut environ 0,3 pour
  tous les modèles et le classement suit le F1. MobileNetV2 en fine-tuning reste premier dans les deux cas.
- Une seule graine : les écarts entre MobileNetV2 et ResNet-18 (+0,012 de F1, +0,080 de frugalité) et entre
  fine-tuning et gelé (+0,05 de F1) sont nettement plus grands que la variabilité attendue.

## Points de vigilance avant la soumission

- Tous ces scores sont mesurés en validation ; le score officiel sera calculé par le serveur sur le test.
- L'essai de JSON du 9 novembre ne prédisait que 2,2 % de `seafood` sur le test, contre 8,5 % en validation :
  regénérer le JSON avec le bon fichier de poids, puis regarder le F1 par classe renvoyé par le serveur (5.2)
  avant la soumission finale.
- La case « update leaderboard » ne doit être cochée qu'une seule fois.

## Décision finale

**Modèle soumis : MobileNetV2 pré-entraîné sur ImageNet, en fine-tuning complet, avec augmentation de données.**

| Élément | Valeur |
|---|---|
| Poids | `checkpoints/mobilenetv2_finetune_best.pth` (époque 15) |
| Paramètres | 2 237 963 (tête à 11 classes : 14 091) |
| Entraînement | SGD avec momentum 0,9, learning rate 1e-3, 15 époques, batch de 64, graine 42 |
| Augmentation | `RandomResizedCrop(224, scale=(0.5, 1.0))` + flip horizontal, normalisation ImageNet |
| F1 macro de validation | 0,925 (accuracy 91,5 %) |
| Frugalité attendue | 0,718 (sur la validation) |

## Actions qui en découlent

| Tâche | Qui | Action |
|---|---|---|
| 5.1 | Maksim | `python predict.py --model mobilenetv2 --mode finetune --weights checkpoints/mobilenetv2_finetune_best.pth --out results/predictions.json`, noter la commande et vérifier la répartition des classes |
| 5.2 | Oggy | Évaluer le JSON sur le serveur sans cocher le leaderboard, relever le F1 global et par classe (surtout `seafood`) |
| 5.3 | Julien | Description de la méthode, sur le modèle de la baseline du prof, à partir du tableau de la décision finale |
| 5.4 | Oggy | Soumission finale unique, après feu vert du groupe |
| 6.4 | Maksim | Ce compte rendu sert de base à la partie « Choix final » du rapport |
