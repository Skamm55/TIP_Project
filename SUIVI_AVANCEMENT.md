# Suivi d'avancement

Journal de ce qui a été fait sur le projet Food-11, par personne et par tâche.
Les numéros de tâche renvoient à la page de suivi « Projet Food-11 ».

Dernière mise à jour : 3 novembre 2026.

## Maksim

### Cadrage du projet (avant la répartition des tâches)

- Lecture de l'énoncé du challenge et du système de notation de TIP (orga.pdf, p.5) : 40 % rapport de TP + bonus de classe inversée, 55 % projet (livrables et présentation), 5 % score de performance.
- Analyse du score de frugalité imposé par le leaderboard et réponse à la question de l'énoncé : le réseau de 11,6 M de paramètres doit atteindre un F1 d'environ 0,843 pour égaler celui de 3,5 M (si *R* est compté en millions, à confirmer par le prof).
- Analyse des supports CVML (lectures et slides complémentaires) et du code Matlab des TD CIFAR-10 et MNIST, pour repérer ce qui pouvait être repris : CNN du TD comme baseline, métriques de la matrice de confusion, courbes d'apprentissage, visualisation des feature maps.
- Choix techniques :
  - Python et PyTorch plutôt que Matlab, pour travailler à plusieurs avec Git ;
  - comparaison de MobileNetV2 et ResNet-18 pré-entraînés, chacun gelé et en fine-tuning, plus la baseline du TD (5 configurations) ;
  - problématique du rapport : quel compromis entre performance et frugalité pour Food-11.
- Recherche d'une machine d'entraînement : PC portable personnel jugé insuffisant (pas de GPU NVIDIA, RAM saturée), choix du PC de Greg (RTX 2000 Ada, 16 Go de VRAM).

### Organisation et gestion du groupe

- Plan d'action complet : 54 tâches réparties en 9 phases, avec les dépendances entre tâches.
- Répartition du travail entre les 4 membres, pour que chacun ait du code, une analyse, une partie du rapport et une partie de la présentation.
- Création de la page de suivi partagée « Projet Food-11 » : déroulé des tâches, une page par personne, page de pilotage avec avancement et temps restant.
- Contact avec le prof (M. Bento) : demande d'un nouveau lien pour l'archive du dataset, expiré sur FileSender, et réponse à son retour.

### Dépôt Git (tâche 0.2)

- Création du dépôt `Skamm55/TIP_Project`.
- Structure initiale : un fichier par module avec son responsable et ce qu'il doit contenir, dossiers `notebooks/` et `results/`, `requirements.txt`, `.gitignore` (données et poids des modèles exclus).
- `README.md` : présentation du projet, rôle de chaque fichier, installation, règles de travail en commun.

### Données (tâches 1.1 à 1.4)

- **1.1** Récupération de l'archive `food-11.zip` sur Moodle et vérification : 13 296 images de train et 3 097 de test, conformes à l'énoncé.
- **1.2** Analyse exploratoire, résumée dans `DATA.md` :
  - 11 classes, de 376 images (`rice`) à 2 000 (`dessert`, `soup`), donc un dataset déséquilibré ;
  - 695 formats d'image différents, médiane 512×512, de 220×207 à 9216×6144 ;
  - figures : répartition des classes et planche d'exemples par classe ;
  - repérage des classes susceptibles d'être confondues (`meat` / `seafood`, `fried` / `meat`, `bread` / `meat`, `dairy` / `dessert`, `egg` / `rice`) et du bruit d'étiquetage.
- **1.3** Écriture de `data.py` :
  - chargement du train par dossiers et du test sans étiquettes (triés numériquement) ;
  - split train / validation stratifié 80 / 20 avec graine fixe (10 636 / 2 660 images), sauvegardé dans `results/split_seed42.csv` ;
  - transformations d'entraînement (recadrage aléatoire, flip) et de validation (redimensionnement, recadrage central) ;
  - option `--subset` pour tester rapidement sur CPU, option `--no-augment` pour l'étude de l'augmentation ;
  - décodage JPEG accéléré par le mode *draft* de PIL (gain d'environ ×2,2 sur les très grandes images) ;
  - script autonome produisant les statistiques et les figures de l'analyse.
- **1.4** Calcul de la normalisation : moyenne RGB de Food-11 (0,570 / 0,451 / 0,327) comparée à celle d'ImageNet (0,485 / 0,456 / 0,406). Décision : garder la normalisation ImageNet pour toutes les configurations, car les réseaux pré-entraînés ont appris leurs poids avec ces valeurs.

### Prochaine tâche

- **1.9** `visualize.py` : visualisation des feature maps de la 1re couche de convolution.

## Greg

Rien de consigné pour l'instant.

## Julien

Rien de consigné pour l'instant.

## Oggy

Rien de consigné pour l'instant.

## Usage de l'IA (pour l'annexe du rapport)

| Qui | Outil | Usage | Parties concernées |
|---|---|---|---|
| Maksim | Claude | Analyse de l'énoncé et des supports de cours, aide au choix de l'approche, plan d'action et organisation | Cadrage, organisation |
| Maksim | Claude Code | Écriture du code à partir de cahiers des charges, puis relecture et tests | `data.py` |
| Maksim | Claude, Claude Code | Rédaction de la note d'analyse à partir des résultats obtenus | `DATA.md` |
