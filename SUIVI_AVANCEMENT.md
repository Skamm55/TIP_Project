# Suivi d'avancement

Journal de ce qui a été fait sur le projet Food-11, par personne et par tâche.
Les numéros de tâche renvoient à la page de suivi « Projet Food-11 ».

Dernière mise à jour : 7 novembre 2026.

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

### Feature maps (tâche 1.9), le 4 novembre

- **1.9** Écriture de `visualize.py`, l'équivalent Python de `Visualize_features.m` du TD :
  - MobileNetV2 ou ResNet-18 pré-entraînés sur ImageNet, activations récupérées avec un forward hook sur la 1re couche (après l'activation par défaut, ou sortie brute de la conv avec `--stage conv`) ;
  - 3 figures : feature maps d'une image (image + 15 canaux), comparaison entre classes (une ligne par classe, 7 canaux les plus actifs), filtres de la 1re couche (32 noyaux 3×3 pour MobileNetV2, 64 noyaux 7×7 pour ResNet-18) ;
  - option `--checkpoint` pour refaire les figures plus tard avec un modèle entraîné (nécessite `models.py`) ;
  - réutilise `data.py` (chargement et prétraitement des images), figures dans `results/figures/`.
- Premières observations pour le rapport (tâche 4.5) : on retrouve des détecteurs de contours et de textures (canaux 4, 5, 7, 19 de MobileNetV2), des canaux sensibles à la couleur et à la luminosité (9, 25, 27), et chez ResNet-18 des filtres en barres orientées (type Gabor) et des filtres de couleurs opposées.
- Relecture de `evaluate.py` et `predict.py` d'Oggy et ajout des docstrings manquantes.

### Assemblage de la chaîne (tâche 2.1), le 7 novembre

- Relecture de `models.py` (Julien) et `train.py` (Greg) après le pull du 6 novembre.
- `models.py` : correct, test OK (baseline 1,08 M paramètres, MobileNetV2 2,24 M, ResNet-18 11,18 M). Seule modif : suppression de l'ancien en-tête, qui faisait une 2e docstring.
- `train.py` réécrit pour que la chaîne marche et pour rester au niveau du cours :
  - 2 bugs corrigés : `get_loaders` n'existe pas dans `data.py` (c'est `get_dataloaders`), et sans `--subset` on passait 0 au lieu de `None`, ce qui ne gardait que 2 images par classe ;
  - options retirées car hors cours ou inutiles : données factices, reprise d'entraînement, précision mixte, weight decay ;
  - gardé : Adam (ou SGD avec momentum), entropie croisée, meilleur modèle choisi sur le F1 macro de validation, courbes d'apprentissage ;
  - learning rate par défaut 1e-3, et 1e-4 en fine-tuning pour ne pas abîmer les poids ImageNet ;
  - en mode frozen, les BatchNorm du réseau gelé restent en mode eval (sinon le réseau « gelé » change quand même) ;
  - à la fin, le meilleur modèle est évalué avec `evaluer()` d'`evaluate.py` : `metrics.json`, matrice de confusion, frugalité, `val_predictions.csv` avec le nom des images ;
  - un dossier par configuration (`results/<modèle>_<mode>/`), pour ne plus écraser les résultats.
- Suppression de `results/run/` : c'était un test fait avec des fausses images et un CNN de secours, pas avec MobileNetV2.
- Test en petit sur CPU (300 images, 2 époques) : baseline, MobileNetV2 gelé et ResNet-18 fine-tuné tournent jusqu'au bout, et `predict.py` relit le `best.pth` produit.
- Commandes des 5 configurations ajoutées dans le README.

### Prochaines tâches

- **1.11** squelette du rapport et journal IA
- **3.3** protocole de l'étude d'augmentation (à envoyer à Greg)
- **4.5** interprétation des feature maps (les figures sont prêtes)
- vérifier avec Greg qu'il comprend la nouvelle version de `train.py` avant les vrais runs

## Greg

Rien de consigné pour l'instant.

### Points à régler

- `train.py` a été réécrit par Maksim le 7 novembre (voir sa section) : à relire avant de lancer les 5 configurations sur ton PC.
- Lancer les 5 configurations avec les commandes du README, et pousser les dossiers `results/<config>/` (sans les `.pth`).

## Julien

Rien de consigné pour l'instant.

### Points à régler

- `count_params` de `models.py` est maintenant utilisé par `train.py`. `evaluate.py` a encore sa propre fonction `compter_params` qui fait la même chose (à voir avec Oggy).

## Oggy

### Évaluation et prédictions (tâches 1.7 et 1.8), le 4 novembre

- **1.7** `evaluate.py` :
  - F1 macro, accuracy, précision / rappel / F1 par classe et matrice de confusion (avec scikit-learn) ;
  - score de frugalité `Frug = P - w / (1 + 1/R)` avec w = 0,3 et R en millions de paramètres. Avec R en millions on retrouve bien les exemples de l'énoncé (0,57 et 0,54), ce qui n'est pas le cas avec R en nombre brut de paramètres ;
  - export de `metrics.json` et de `val_predictions.csv` (image, vraie classe, classe prédite, confiance), utile pour l'analyse des erreurs ;
  - auto-test sans dataset (`python evaluate.py`) : 5 tests, tous OK.
- **1.8** `predict.py` :
  - génère le JSON pour le serveur au format `{"0": "bread", ...}`, trié dans l'ordre numérique ;
  - vérifie le JSON avant de l'écrire : 3 097 images, clés sans extension, pas de doublon, classes connues ;
  - accepte des poids enregistrés seuls ou dans un checkpoint (sous la clé `"model"`) ;
  - auto-test sans dataset (`python predict.py --self-test`) : 4 tests, tous OK.

### Points à caler avec les autres

- Greg (`train.py`) : le checkpoint doit contenir les poids sous la clé `"model"` (ou être un state_dict seul), sinon `predict.py` ne pourra pas le relire.
- Greg : `evaluer()` écrit par défaut dans `results/` ; il faudra lui passer un dossier par configuration (`dossier=results/<config>`) pour ne pas écraser les résultats d'un modèle par ceux du suivant.
- Julien (`models.py`) : `predict.py` appelle `get_model(name, mode, num_classes=11)`, comme prévu dans l'en-tête de `models.py`. `evaluate.py` a sa propre fonction `compter_params`, qui fait la même chose que le `count_params` prévu dans `models.py` : on pourra n'en garder qu'une.

## Usage de l'IA (pour l'annexe du rapport)

| Qui    | Outil               | Usage                                                                                                    | Parties concernées          |
| ------ | ------------------- | -------------------------------------------------------------------------------------------------------- | --------------------------- |
| Maksim | Claude              | Analyse de l'énoncé et des supports de cours, aide au choix de l'approche, plan d'action et organisation | Cadrage, organisation       |
| Maksim | Claude Code         | Écriture du code à partir de cahiers des charges, puis relecture et tests                                | `data.py`                   |
| Maksim | Claude, Claude Code | Rédaction de la note d'analyse à partir des résultats obtenus                                            | `DATA.md`                   |
| Maksim | Claude Code         | Écriture du code à partir d'un cahier des charges, tests et vérification des figures                     | `visualize.py`              |
| Oggy   | Claude Code         | Aide à l'écriture du code (commits co-signés par Claude)                                                 | `evaluate.py`, `predict.py` |
| Maksim | Claude Code         | Relecture de `models.py` et `train.py`, correction des bugs et réécriture de `train.py`, tests           | `train.py`, `models.py`     |
