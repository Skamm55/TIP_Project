# TIP_Project : challenge Food-11

Projet de TIP (4TC, INSA Lyon). On classe des images de nourriture en 11 catégories avec du transfer learning.
Le but est d'avoir un bon F1 macro tout en gardant un modèle léger (peu de paramètres).

## Fichiers

| Fichier                   | Rôle                                                       | Qui    |
| ------------------------- | ---------------------------------------------------------- | ------ |
| `data.py`                 | chargement des données, split, transformations             | Maksim |
| `models.py`               | baseline du TD, MobileNetV2, ResNet-18                     | Julien |
| `train.py`                | entraînement                                               | Greg   |
| `evaluate.py`             | F1, matrice de confusion, frugalité                        | Oggy   |
| `predict.py`              | JSON des prédictions sur le test                           | Oggy   |
| `visualize.py`            | feature maps et filtres de la 1re couche                   | Maksim |
| `analyse_entrainement.py` | courbes, biais / variance, étude de l'augmentation         | Maksim |
| `frugalite.py`            | tableau de frugalité et figure F1 / paramètres             | Maksim |
| `analyse_erreurs.py`      | matrice de confusion et erreurs du meilleur modèle         | Oggy   |
| `results/`                | métriques des runs, tableaux et figures                    | tous   |

Les documents :

| Fichier                     | Contenu                                                     |
| --------------------------- | ----------------------------------------------------------- |
| `DATA.md`                   | analyse du dataset et normalisation                         |
| `PROTOCOLE_AUGMENTATION.md` | protocole de l'étude avec / sans augmentation               |
| `DECISION_MODELE_FINAL.md`  | choix du modèle soumis, justifié étape par étape            |
| `SUIVI_AVANCEMENT.md`       | journal de chacun et tableau d'usage de l'IA                |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # sous Windows
pip install -r requirements.txt
```

Sur un PC avec une carte NVIDIA, il faut d'abord installer PyTorch avec CUDA (la commande est sur pytorch.org), et ensuite seulement `requirements.txt`.

## Données

Dézipper `food-11.zip` (donné par le prof) dans `data/food-11/`.
Le dossier `data/` ne doit jamais aller sur Git.

## Commandes

Tests sans dataset (quelques secondes, rien à télécharger) :

```bash
python models.py              # les 6 modèles donnent une sortie de taille (2, 11)
python evaluate.py            # F1, frugalité (exemples de l'énoncé), export des fichiers
python predict.py --self-test # format du JSON
```

Avec le dataset, vérification du chargement puis test rapide de toute la chaîne (sur CPU, quelques minutes) :

```bash
python data.py --subset 300
python train.py --model mobilenetv2 --mode frozen --subset 300 --epochs 1 --out-dir results/test_chaine
```

Le test écrit dans `results/test_chaine/`, qu'on peut supprimer ensuite.

Les 5 configurations (sur le PC de Greg) :

```bash
python train.py --model baseline
python train.py --model mobilenetv2 --mode frozen
python train.py --model mobilenetv2 --mode finetune --epochs 15
python train.py --model resnet18 --mode frozen
python train.py --model resnet18 --mode finetune --epochs 15
```

Le fine-tuning se fait sur 15 époques, comme prévu dans `PROTOCOLE_AUGMENTATION.md` (le run sans
augmentation de la tâche 3.4 doit avoir exactement les mêmes réglages).

Chaque run écrit dans `results/<modèle>_<mode>/` : courbes, `history.csv`, `metrics.json`
(F1, matrice de confusion, frugalité), `val_predictions.csv` et `best.pth` (pas sur Git).
Pour l'étude de l'augmentation, ajouter `--no-augment` (le dossier finit par `_noaug`).

Réglages par défaut, valeurs de départ à ajuster selon les courbes :

| Configuration                 | Optimiseur            | Learning rate |
| ----------------------------- | --------------------- | ------------- |
| baseline                      | SGD, momentum 0.9     | 1e-4          |
| pré-entraîné, mode frozen     | SGD, momentum 0.9     | 1e-2          |
| pré-entraîné, mode finetune   | SGD, momentum 0.9     | 1e-3          |

SGD avec momentum 0.9, c'est le `sgdm` des `trainingOptions` du TD Matlab.
Pour la baseline, le TD utilisait 1e-3, mais sur des images 224×224 la couche FC est
beaucoup plus grosse (8×111×111 entrées au lieu de 8×15×15) et la loss diverge avec 1e-3.
On peut tout changer avec `--lr`, `--optimizer adam`, `--momentum` et `--weight-decay`.

Analyses, à partir des fichiers de `results/` (pas besoin des poids ni du GPU) :

```bash
python analyse_entrainement.py   # courbes, synthese_runs.csv, figures avec / sans augmentation
python frugalite.py              # frugalite.csv et f1_vs_parametres.png
python analyse_erreurs.py        # matrice de confusion et scores par classe du modèle final
```

Données et feature maps (dataset nécessaire, poids ImageNet téléchargés au 1er lancement) :

```bash
python data.py --plot-counts --samples --sizes --stats       # chiffres et figures de DATA.md
python visualize.py --model resnet18 --filters               # filtres de la 1re couche
python visualize.py --classes bread dessert meat seafood soup rice
```

JSON pour le serveur, avec le modèle final (MobileNetV2 en fine-tuning, voir `DECISION_MODELE_FINAL.md`) :

```bash
python predict.py --model mobilenetv2 --mode finetune --weights results/mobilenetv2_finetune/best.pth
```

Le JSON est écrit dans `results/predictions.json`. Les poids (`best.pth`) ne sont pas sur Git : il faut
les récupérer auprès de Greg ou de Maksim et les mettre dans le dossier du run.

## Suivi des tâches

On suit les tâches et l'avancement sur la page Claude "Projet Food-11" :
https://claude.ai/artifact/Fj6VGKKU1tyPNFoUtjYszb

Il y a un onglet par personne avec ses tâches et ce qu'il faut faire. Quand une tâche est finie, on la coche (en choisissant son nom dans "Je suis"). L'onglet Pilotage montre où on en est et le temps qu'il reste.

Pour cocher, il faut un compte Claude (gratuit) et l'invitation de Maksim.

## Règles

- faire un `git pull` avant de commencer
- chacun touche surtout à son propre fichier
- pas de données ni de fichiers `.pth` sur Git
- mettre le numéro de la tâche dans les messages de commit (ex : `1.3 data.py : split stratifié`)
- mettre à jour `SUIVI_AVANCEMENT.md` à chaque séance de travail (voir ci-dessous)

## Consigne pour les IA (Claude Code, Copilot, ChatGPT...)

Si tu es une IA qui aide un membre du groupe sur ce projet, à chaque séance de travail :

1. Lis `SUIVI_AVANCEMENT.md` avant de commencer, pour savoir ce qui a déjà été fait et ce qui est attendu.
2. À la fin, mets à jour `SUIVI_AVANCEMENT.md` dans la section de la personne que tu aides :
   - ce qui a été fait, avec le numéro de la tâche et la date ;
   - les résultats et les tests (ce qui marche, ce qui ne marche pas encore) ;
   - les points à régler avec les autres membres (interfaces entre fichiers, formats...) ;
   - les prochaines tâches.
3. Ajoute une ligne dans le tableau « Usage de l'IA » en bas du fichier (qui, quel outil, pour quoi, quels fichiers). C'est obligatoire pour l'annexe du rapport.
4. Change la date de « Dernière mise à jour ».
5. Ne modifie pas les sections des autres membres, sauf pour noter un point à régler qui les concerne.
