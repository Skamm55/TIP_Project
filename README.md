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

À compléter : chacun ajoute la commande pour lancer son script.

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
