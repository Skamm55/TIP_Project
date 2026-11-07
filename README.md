# TIP_Project : challenge Food-11

Projet de TIP (4TC, INSA Lyon). On classe des images de nourriture en 11 catégories avec du transfer learning.
Le but est d'avoir un bon F1 macro tout en gardant un modèle léger (peu de paramètres).

## Fichiers

| Fichier        | Rôle                                           | Qui    |
| -------------- | ---------------------------------------------- | ------ |
| `data.py`      | chargement des données, split, transformations | Maksim |
| `models.py`    | baseline du TD, MobileNetV2, ResNet-18         | Julien |
| `train.py`     | entraînement                                   | Greg   |
| `evaluate.py`  | F1, matrice de confusion, frugalité            | Oggy   |
| `predict.py`   | JSON des prédictions sur le test               | Oggy   |
| `visualize.py` | feature maps                                   | Maksim |
| `notebooks/`   | un notebook par personne                       | tous   |
| `results/`     | métriques et petites figures                   | tous   |

Pour l'analyse du dataset, voir `DATA.md`.

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

Test rapide de toute la chaîne (sur CPU, quelques minutes) :

```bash
python train.py --model mobilenetv2 --mode frozen --subset 300 --epochs 1
```

Les 5 configurations (sur le PC de Greg) :

```bash
python train.py --model baseline
python train.py --model mobilenetv2 --mode frozen
python train.py --model mobilenetv2 --mode finetune
python train.py --model resnet18 --mode frozen
python train.py --model resnet18 --mode finetune
```

Chaque run écrit dans `results/<modèle>_<mode>/` : courbes, `history.csv`, `metrics.json`
(F1, matrice de confusion, frugalité), `val_predictions.csv` et `best.pth` (pas sur Git).
Le learning rate vaut 1e-3 par défaut et 1e-4 en fine-tuning (`--lr` pour le changer).
Pour l'étude de l'augmentation, ajouter `--no-augment` (le dossier finit par `_noaug`).

JSON pour le serveur, à partir d'un modèle entraîné :

```bash
python predict.py --model mobilenetv2 --mode frozen --weights results/mobilenetv2_frozen/best.pth
```

Feature maps : `python visualize.py --model mobilenetv2` (voir `python visualize.py -h`).

## Suivi des tâches

On suit les tâches et l'avancement sur la page Claude "Projet Food-11" :
https://claude.ai/artifact/Fj6VGKKU1tyPNFoUtjYszb

Il y a un onglet par personne avec ses tâches et ce qu'il faut faire. Quand une tâche est finie, on la coche (en choisissant son nom dans "Je suis"). L'onglet Pilotage montre où on en est et le temps qu'il reste.

Pour cocher, il faut un compte Claude (gratuit) et l'invitation de Maksim.

## Règles

- faire un `git pull` avant de commencer
- chacun touche surtout à son propre fichier
- un notebook par personne
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
