# Challenge Food-11 : classification d'images de nourriture

Projet de TIP (4TC, INSA Lyon, 2026-2027), par Maksim Nèples, Greg Blanzat, Julien Chautant et
Ognjen Stojanovic.

Le but du challenge est de classer des photos de plats en 11 catégories (bread, dairy, dessert, egg,
fried, meat, noodles-pasta, rice, seafood, soup, vegetable-fruit). Le classement ne se fait pas sur le
F1 seul : le score de frugalité pénalise les réseaux qui ont beaucoup de paramètres,

    Frug = F1 macro − 0,3 · R / (R + 1)      avec R le nombre de paramètres en millions.

On a comparé le CNN du TD, entraîné de zéro, à deux réseaux pré-entraînés sur ImageNet (MobileNetV2 et
ResNet-18), chacun en mode gelé et en fine-tuning. Le modèle qu'on a retenu est **MobileNetV2 en
fine-tuning complet, avec augmentation de données** : F1 macro de 0,925 sur notre validation, et 0,926
sur le test du serveur, pour une frugalité de 0,719.

Toute la démarche et l'analyse des résultats sont dans le rapport. Ce README explique comment est
organisé le code et comment tout relancer.

## Ce qu'il y a dans le dossier

```
data.py                    chargement des images, split train / validation, transformations
models.py                  les trois réseaux (baseline du TD, MobileNetV2, ResNet-18)
train.py                   entraînement d'une configuration
evaluate.py                F1 macro, scores par classe, matrice de confusion, frugalité
predict.py                 prédictions sur le test et fichier JSON pour le serveur
analyse_entrainement.py    courbes d'apprentissage et comparaison avec / sans augmentation
frugalite.py               tableau de frugalité des runs
analyse_erreurs.py         matrice de confusion et erreurs du modèle retenu
visualize.py               filtres et feature maps de la 1re couche de convolution
requirements.txt           bibliothèques Python nécessaires

results/
    baseline/                      un dossier par configuration entraînée :
    mobilenetv2_frozen/              config.json   réglages du run
    mobilenetv2_finetune/            history.csv   loss, accuracy et F1 à chaque époque
    mobilenetv2_finetune_noaug/      metrics.json  scores du meilleur modèle sur la validation
    resnet18_frozen/
    resnet18_finetune/
    test_surapprentissage/         vérification de la chaîne sur 41 images
    figures/                       toutes les figures du rapport
    synthese_runs.csv              chiffres clés de chaque run
    frugalite.csv                  F1, nombre de paramètres et frugalité de chaque run
    predictions.json               le fichier soumis au serveur
```

Le dossier du modèle retenu contient en plus `val_predictions.csv` (la prédiction pour chaque image de
validation), qui sert à l'analyse des erreurs.

Deux choses ne sont pas dans le zip, comme demandé dans les consignes : les images du dataset et les
poids des réseaux entraînés (fichiers `best.pth`, entre 4 et 45 Mo chacun). Les scripts d'analyse
n'en ont pas besoin, mais pour refaire un JSON il faut d'abord ré-entraîner le modèle (voir plus bas).

## Installation

Il faut Python 3.10 ou plus récent.

```bash
python -m venv .venv
.venv\Scripts\activate          # sous Windows (source .venv/bin/activate sous Linux / Mac)
pip install -r requirements.txt
```

Pour entraîner sur une carte NVIDIA, installer d'abord PyTorch avec CUDA en suivant la commande donnée
sur pytorch.org, puis seulement `requirements.txt`. Sans GPU tout marche aussi, mais l'entraînement
complet devient très long.

Le dataset Food-11 fourni pour le cours se dézippe dans `data/food-11/`, ce qui doit donner :

```
data/food-11/train/bread/...          un dossier par classe, 13 296 images en tout
data/food-11/test/0.jpg ... 3096.jpg  3 097 images sans étiquette
```

## Vérifier que tout fonctionne

Ces trois tests ne demandent ni le dataset ni de téléchargement, ils prennent quelques secondes :

```bash
python models.py               # chaque réseau doit sortir un vecteur de 11 scores
python evaluate.py             # F1 et frugalité, dont les deux exemples de l'énoncé (0,57 et 0,54)
python predict.py --self-test  # format du JSON attendu par le serveur
```

Avec le dataset en place, on peut faire tourner toute la chaîne en petit, sur 300 images et une seule
époque (quelques minutes sur un processeur) :

```bash
python data.py --subset 300
python train.py --model mobilenetv2 --mode frozen --subset 300 --epochs 1 --out-dir results/test_chaine
```

Le dossier `results/test_chaine/` peut être supprimé ensuite.

## Ré-entraîner les modèles

Ce sont les six commandes qu'on a lancées. Les temps indiqués sont ceux mesurés sur notre GPU
(RTX 2000 Ada) :

```bash
python train.py --model baseline --epochs 20                          # environ 7 min
python train.py --model mobilenetv2 --mode frozen                     # environ 4 min
python train.py --model resnet18 --mode frozen                        # environ 4 min
python train.py --model mobilenetv2 --mode finetune --epochs 15       # environ 12 min
python train.py --model resnet18 --mode finetune --epochs 15          # environ 8 min
python train.py --model mobilenetv2 --mode finetune --epochs 15 --no-augment
```

La dernière commande est la même que la quatrième, sans l'augmentation de données : c'est la
comparaison faite dans le rapport.

Par défaut, l'optimiseur est SGD avec un momentum de 0,9 (le `sgdm` des TD Matlab), avec un batch de 64
et la graine 42. Le learning rate dépend de la configuration :

| Configuration                  | Learning rate |
| ------------------------------ | ------------- |
| baseline                       | 1e-4          |
| réseau pré-entraîné, gelé      | 1e-2          |
| réseau pré-entraîné, fine-tuné | 1e-3          |

Pour la baseline, on a dû descendre à 1e-4 : avec des images de 224×224, sa couche entièrement
connectée reçoit 8×111×111 valeurs (contre 8×15×15 dans le TD sur CIFAR-10), et la loss diverge à 1e-3.
Tous ces réglages peuvent être changés avec `--lr`, `--optimizer adam`, `--momentum`, `--weight-decay`
ou `--batch-size` (voir `python train.py -h`).

Chaque run écrit ses résultats dans `results/<réseau>_<mode>/`. Le meilleur modèle, celui qui a le
meilleur F1 macro de validation, est enregistré dans `best.pth`, puis évalué automatiquement.

## Générer le fichier pour le serveur

Une fois le modèle retenu ré-entraîné :

```bash
python predict.py --model mobilenetv2 --mode finetune --weights results/mobilenetv2_finetune/best.pth
```

Le script prédit une classe pour chacune des 3 097 images du test, vérifie le format puis écrit
`results/predictions.json`, sous la forme `{"0": "soup", "1": "bread", ...}`. Comptez une dizaine de
minutes sur un processeur, beaucoup moins sur un GPU.

## Refaire les figures du rapport

Les figures de `results/figures/` se regénèrent avec les commandes suivantes. Les trois scripts
d'analyse travaillent uniquement à partir des fichiers de `results/`, sans dataset ni poids.

| Figure                                | Commande                                                | Dataset |
| ------------------------------------- | ------------------------------------------------------- | ------- |
| `class_counts.png`                    | `python data.py --plot-counts`                          | oui     |
| `samples_per_class.png`               | `python data.py --samples`                              | oui     |
| `f1_validation_runs.png`              | `python analyse_entrainement.py`                        | non     |
| `train_vs_val_finetuning.png`         | `python analyse_entrainement.py`                        | non     |
| `train_vs_val_gele.png`               | `python analyse_entrainement.py`                        | non     |
| `augmentation_avec_sans.png`          | `python analyse_entrainement.py`                        | non     |
| `augmentation_f1_par_classe.png`      | `python analyse_entrainement.py`                        | non     |
| `f1_vs_parametres.png`                | `python frugalite.py`                                   | non     |
| `confusion_mobilenetv2_finetune.png`  | `python analyse_erreurs.py`                             | non     |
| `erreurs_mobilenetv2_finetune.png`    | `python analyse_erreurs.py --images "..."` (ci-dessous) | oui     |
| `filters_resnet18.png`                | `python visualize.py --model resnet18 --filters`        | non     |
| `featuremaps_mobilenetv2_classes.png` | `python visualize.py --classes ...` (ci-dessous)        | oui     |

Les deux commandes un peu longues, avec les images qu'on a choisies :

```bash
python analyse_erreurs.py --images "meat/579.jpg,fried/705.jpg,dairy/538.jpg,fried/225.jpg,dessert/532.jpg,bread/465.jpg"
python visualize.py --classes bread dessert meat seafood soup rice
```

`visualize.py` télécharge les poids ImageNet au premier lancement (14 Mo pour MobileNetV2, 45 Mo pour
ResNet-18).

`python data.py --sizes --stats` affiche en plus les tailles d'images et les moyennes et écarts-types
RGB du dataset, comparés à ceux d'ImageNet.
