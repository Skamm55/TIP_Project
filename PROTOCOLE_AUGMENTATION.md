# Protocole de l'étude de l'augmentation de données (tâche 3.3)

## Objectif

Mesurer l'effet de l'augmentation de données sur la généralisation du modèle.
L'augmentation est le remède proposé en cas de variance élevée, c'est-à-dire de surapprentissage
(complementary slides, p.28).

## Configuration étudiée

La meilleure configuration en fine-tuning à l'issue de la tâche 3.2 (MobileNetV2 ou ResNet-18,
selon le F1 macro de validation) : *à compléter après la 3.2*.

On choisit le fine-tuning plutôt que le mode gelé parce que plusieurs millions de paramètres y sont
entraînés, contre 14 091 (MobileNetV2) ou 5 643 (ResNet-18) en mode gelé : c'est en fine-tuning que
le risque de surapprentissage, et donc l'effet de l'augmentation, est le plus important.

## Les deux versions comparées

- Avec augmentation (run de la 3.2) : `RandomResizedCrop(224, scale=(0.5, 1.0))` et flip horizontal.
- Sans augmentation (run de la 3.4, option `--no-augment`) : `Resize(256)` puis `CenterCrop(224)`,
  les mêmes transformations que pour la validation. Le modèle voit donc exactement les mêmes images
  à chaque époque.

Seule l'option `--no-augment` change entre les deux runs. Le modèle, le mode, l'optimiseur
(SGD avec momentum 0,9), le learning rate, le nombre d'époques (15), la graine (42) et le split
train / validation sont identiques.

## Choix des augmentations

- Recadrage aléatoire : les photos de plats sont cadrées de façons très variées (plat entier,
  gros plan, assiette coupée). Le modèle doit reconnaître l'aliment quels que soient le zoom et la position.
- Flip horizontal : un plat retourné de gauche à droite reste le même plat, l'étiquette ne change pas.
- Pas de modification des couleurs : la couleur est une information discriminante pour les aliments
  (riz blanc, sauce tomate, légumes verts), et l'analyse des données a montré des couleurs propres
  au dataset (plus de rouge et moins de bleu qu'ImageNet, voir `DATA.md`).

## Mesures (tâche 4.4)

| Mesure                                                        | Source                        | Attendu sans augmentation                   |
| ------------------------------------------------------------- | ----------------------------- | ------------------------------------------- |
| F1 macro de validation                                        | `config.json`, `metrics.json` | plus faible                                 |
| Écart entre accuracy train et validation (dernière époque)    | `history.csv`                 | plus grand                                  |
| Loss de validation au fil des époques                         | `curves.png`, `history.csv`   | remonte plus tôt                            |
| F1 par classe, surtout `rice`, `dairy` et `noodles-pasta`     | `metrics.json`                | baisse plus marquée sur les petites classes |

## Hypothèse

Sans augmentation, le modèle atteint rapidement une accuracy train proche de 100 %
mais généralise moins bien. L'effet devrait être plus fort sur les petites classes
(`rice` 376 images, `dairy` 573, `noodles-pasta` 587), qui disposent de moins d'images différentes.

## Limite

Une seule graine est utilisée. Un écart de F1 de l'ordre de 0,01 peut venir du hasard :
seul un écart net, cohérent avec les courbes d'apprentissage, sera interprété.
Les petites classes n'ont que 75 à 117 images de validation, donc leur F1 bouge facilement.

## Commandes pour Greg

Le run avec augmentation est celui de la 3.2 (il doit être lancé avec `--epochs 15`).
Le run sans augmentation, après la 3.2 (exemple si ResNet-18 l'emporte) :

```bash
python train.py --model resnet18 --mode finetune --epochs 15 --no-augment
```

Les résultats vont dans `results/resnet18_finetune_noaug/`, à côté de `results/resnet18_finetune/`.
