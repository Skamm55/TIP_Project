# TIP_Project : challenge de classification Food-11

Projet de TIP (4TC, INSA Lyon) : classification d'images en 11 catégories d'aliments par transfer learning,
avec un compromis entre performance (F1 macro) et frugalité (nombre de paramètres).

## Structure

| Fichier | Rôle | Responsable |
|---|---|---|
| `data.py` | chargement, split, transformations | Maksim |
| `models.py` | baseline du TD, MobileNetV2, ResNet-18 | Julien |
| `train.py` | entraînement d'une configuration | Greg |
| `evaluate.py` | F1, confusion, frugalité | Oggy |
| `predict.py` | JSON de prédictions du test | Oggy |
| `visualize.py` | feature maps | Maksim |
| `notebooks/` | analyses (un notebook par personne) | tous |
| `results/` | métriques et figures légères | tous |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

PC avec GPU NVIDIA : installer d'abord PyTorch avec CUDA (commande sur pytorch.org), puis `requirements.txt`.

## Données

Dézipper `food-11.zip` (fourni par l'enseignant) dans `data/food-11/`.
Le dossier `data/` n'est jamais envoyé sur Git.

## Commandes

À compléter par chacun avec la commande de son script.

## Suivi des tâches

Le déroulé du projet, les tâches de chacun et l'avancement sont suivis sur la page Claude **Projet Food-11** :
https://claude.ai/artifact/Fj6VGKKU1tyPNFoUtjYszb

- un onglet par personne avec ses tâches et ce qu'il faut faire concrètement
- cocher une tâche quand elle est terminée (choisir son nom dans « Je suis »)
- l'onglet Pilotage montre l'avancement et le temps restant

Il faut un compte Claude (gratuit) et l'invitation de Maksim pour pouvoir cocher.

## Règles de travail

- `git pull` avant de commencer
- chacun modifie surtout son propre fichier
- un notebook par personne
- jamais de données ni de fichiers `.pth` sur Git
- dans les messages de commit, citer le numéro de la tâche (ex. `1.3 data.py : split stratifié`)
