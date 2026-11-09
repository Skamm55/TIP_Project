# Suivi d'avancement

Journal de ce qui a été fait sur le projet Food-11, par personne et par tâche.
Les numéros de tâche renvoient à la page de suivi « Projet Food-11 ».

Dernière mise à jour : 9 novembre 2026.

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
  - options retirées car hors cours ou inutiles : données factices, reprise d'entraînement, précision mixte ;
  - gardé : entropie croisée, meilleur modèle choisi sur le F1 macro de validation, courbes d'apprentissage ;
  - en mode frozen, les BatchNorm du réseau gelé restent en mode eval (sinon le réseau « gelé » change quand même) ;
  - à la fin, le meilleur modèle est évalué avec `evaluer()` d'`evaluate.py` : `metrics.json`, matrice de confusion, frugalité, `val_predictions.csv` avec le nom des images ;
  - un dossier par configuration (`results/<modèle>_<mode>/`), pour ne plus écraser les résultats.
- Suppression de `results/run/` : c'était un test fait avec des fausses images et un CNN de secours, pas avec MobileNetV2.
- Test en petit sur CPU (300 images, 2 époques) : baseline, MobileNetV2 gelé et ResNet-18 fine-tuné tournent jusqu'au bout, et `predict.py` relit le `best.pth` produit.
- Commandes des 5 configurations ajoutées dans le README.
- Optimiseur aligné sur le TD : SGD avec momentum 0.9 par défaut (le `sgdm` des `trainingOptions` Matlab), Adam en option, plus `--momentum` et `--weight-decay`. Learning rate par défaut : 1e-4 pour la baseline, 1e-2 en mode frozen, 1e-3 en fine-tuning.
- Diagnostic de la loss de la baseline (30 à 50 dans le premier test). Sur les 20 premières itérations (300 images, batch 64) :
  - loss au 1er batch = 2,42, soit ln(11) : l'initialisation et la normalisation des entrées sont correctes ;
  - SGD lr 1e-3 : 2,42 puis montée jusqu'à 14,7 ; Adam lr 1e-3 : 2,42 puis 39 dès la 2e itération et jusqu'à 60 → divergence due au pas d'apprentissage ;
  - SGD lr 1e-4 : descente régulière de 2,42 à environ 1,9 ; Adam lr 1e-4 : pic à 4,6 puis descente vers 2,0 ;
  - explication pour le rapport : la FC de la baseline reçoit 8×111×111 = 98 568 valeurs en 224×224, contre 8×15×15 = 1 800 sur CIFAR-10 dans le TD, donc le lr de 1e-3 du TD est trop fort ici. L'architecture de la baseline n'est pas modifiée.

### Test de surapprentissage (tâche 2.2) et protocole d'augmentation (tâche 3.3), le 7 novembre

- **2.2** MobileNetV2 fine-tuné sur 41 images, sans augmentation, 30 époques, batch 8 (`results/test_surapprentissage/`) :
  - accuracy train de 0,02 à 1,0 (puis 0,976, une seule image ratée), loss train de 2,49 à environ 0,1 : la chaîne sait apprendre, pas de bug ;
  - loss de validation minimale à l'époque 12 (1,71) puis remonte au-dessus de 2,2 : surapprentissage bien visible, figure utilisable pour la partie biais / variance.
- **3.3** Protocole écrit dans `PROTOCOLE_AUGMENTATION.md` : meilleure config en fine-tuning, avec / sans `--no-augment`, 15 époques, seule l'augmentation change. Les commandes de fine-tuning du README sont passées à `--epochs 15` pour que le run de la 3.2 serve de référence.

### Courbes d'apprentissage et diagnostic biais / variance (tâche 4.2)

Figures : `results/figures/f1_validation_runs.png`, `train_vs_val_finetuning.png`, `train_vs_val_gele.png`.
Synthèse chiffrée : `results/synthese_runs.csv`.

| Run | Acc. train (dernière époque) | Acc. validation | Écart | Diagnostic |
|---|---|---|---|---|
| Baseline (TD) | 37,6 % | 36,4 % | 1,2 pt | biais élevé |
| MobileNetV2 gelé | 84,7 % | 86,2 % | −1,6 pt | biais modéré, pas de variance |
| ResNet-18 gelé | 85,0 % | 84,6 % | 0,4 pt | biais modéré, pas de variance |
| MobileNetV2 fine-tuning | 97,8 % | 91,5 % | 6,2 pts | début de variance |
| ResNet-18 fine-tuning | 97,3 % | 90,5 % | 6,9 pts | début de variance |

Le diagnostic suit la grille du cours (complementary slides, p.25) : un écart faible entre train et
validation avec des performances basses traduit un biais élevé (sous-apprentissage), un écart important
traduit une variance élevée (surapprentissage).

**Baseline.** La loss d'entraînement ne descend que de 2,17 à 1,82 en 20 époques, et les accuracies train
et validation restent proches de 37 %. Le modèle n'apprend même pas correctement ses données d'entraînement :
c'est un biais élevé. Une seule couche de 8 filtres 3×3 ne suffit pas à distinguer 11 catégories d'aliments.
Les remèdes proposés par le cours en cas de biais élevé (complementary slides, p.28) sont d'utiliser un réseau
plus gros ou de changer d'architecture, ce que fait précisément le transfer learning.

**Modèles gelés.** Le F1 de validation plafonne entre 0,86 et 0,87 dès les époques 3 à 8 : entraîner plus
longtemps n'apporterait presque rien. Il n'y a pas de surapprentissage (écart train / validation négligeable).
La limite vient des features ImageNet figées, puisque seule la dernière couche est entraînée (14 091 paramètres
pour MobileNetV2, 5 643 pour ResNet-18). L'accuracy de validation dépasse même légèrement celle du train pour
MobileNetV2 : l'accuracy train est mesurée pendant l'entraînement, avec l'augmentation de données et le dropout
actifs, ce qui la pénalise. MobileNetV2 gelé présente un décrochage ponctuel à l'époque 6 (accuracy de
validation de 0,80, loss de 0,64), signe d'une oscillation de l'apprentissage de la tête avec un learning rate
de 1e-2 ; le run se stabilise dès l'époque suivante. Le remède est de dégeler le réseau, c'est-à-dire de passer
au fine-tuning.

**Modèles en fine-tuning.** Ce sont les meilleurs modèles (F1 de validation de 0,9252 pour MobileNetV2 et
0,9134 pour ResNet-18), mais l'écart train / validation atteint 6 à 7 points. La loss de validation atteint son
minimum à l'époque 11 pour MobileNetV2 (0,290) et à l'époque 13 pour ResNet-18 (0,303), puis remonte (0,308 et
0,312 à l'époque 15) alors que la loss d'entraînement continue de baisser : c'est le début du surapprentissage.
Le F1 progresse encore, mais de seulement 0,005 sur les quatre dernières époques. Le meilleur modèle tombe en
fin d'entraînement, mais les courbes montrent un plateau et non un modèle inachevé : augmenter le nombre
d'époques accentuerait surtout la variance. Les remèdes du cours en cas de variance élevée sont l'augmentation
de données et la régularisation (complementary slides, p.28) ; l'effet de l'augmentation est mesuré dans
l'étude dédiée (tâche 4.4).

**Temps de calcul.** Une époque de fine-tuning dure environ 42 s pour MobileNetV2 contre 27 s pour ResNet-18,
alors que MobileNetV2 a cinq fois moins de paramètres (2,24 M contre 11,18 M). Ses convolutions en profondeur
(depthwise) réduisent le nombre de paramètres mais sont moins efficacement exécutées sur GPU. Le score de
frugalité du challenge ne compte que les paramètres : un modèle frugal au sens du challenge n'est pas
forcément le plus rapide à entraîner.

### Tableau de frugalité et choix du modèle final (tâche 4.1)

Script : `frugalite.py`. Résultats : `results/frugalite.csv`, figure `results/figures/f1_vs_parametres.png`.

Score de frugalité du challenge : Frug = F1 − w·R / (R + 1), avec w = 0,3 et R le nombre de paramètres
en millions. Exemple pour MobileNetV2 en fine-tuning : R = 2,238, pénalité = 0,3 × 2,238 / 3,238 = 0,207,
Frug = 0,9252 − 0,207 = 0,718.

| Run | F1 validation | Paramètres | Pénalité | Frugalité |
|---|---|---|---|---|
| Baseline (TD) | 0,3346 | 1,08 M | 0,156 | 0,178 |
| MobileNetV2 gelé | 0,8740 | 2,24 M | 0,207 | 0,667 |
| ResNet-18 gelé | 0,8589 | 11,18 M | 0,275 | 0,583 |
| **MobileNetV2 fine-tuning** | **0,9252** | **2,24 M** | **0,207** | **0,718** |
| ResNet-18 fine-tuning | 0,9134 | 11,18 M | 0,275 | 0,638 |
| MobileNetV2 fine-tuning sans augmentation | 0,9108 | 2,24 M | 0,207 | 0,703 |

**Choix proposé : MobileNetV2 en fine-tuning, avec augmentation de données.** Il obtient à la fois le
meilleur F1 de validation et le meilleur score de frugalité.

**Le choix du réseau pèse plus que la stratégie d'entraînement.** MobileNetV2 gelé (0,667) devance
ResNet-18 en fine-tuning (0,638) alors que son F1 est nettement plus faible (0,874 contre 0,913) : comme
dans l'exemple de l'énoncé, un réseau léger un peu moins précis passe devant un réseau plus lourd et plus
précis. Pour égaler MobileNetV2 en fine-tuning, ResNet-18 devrait compenser une pénalité supérieure de
0,275 − 0,207 = 0,068, soit atteindre un F1 d'au moins 0,925 + 0,068 ≈ 0,993, ce qui est irréaliste.

**Geler un réseau ne réduit pas sa pénalité.** Le challenge compte tous les paramètres du réseau, y compris
ceux qui ne sont pas entraînés : MobileNetV2 gelé et en fine-tuning ont la même pénalité et ne sont
départagés que par leur F1.

**La légèreté seule ne suffit pas.** La baseline a le moins de paramètres (1,08 M) mais son F1 est si faible
qu'elle obtient la pire frugalité.

**Le choix ne dépend pas de l'unité de R.** Si R était compté en nombre brut de paramètres, la pénalité
vaudrait environ 0,3 pour tous les modèles et le classement serait celui du F1 : MobileNetV2 en
fine-tuning resterait premier.

**Limite.** Ces scores sont calculés sur la validation ; le score officiel sera calculé par le serveur sur
le jeu de test.

### Prochaines tâches

- **1.11** squelette du rapport et journal IA
- **4.5** interprétation des feature maps (les figures sont prêtes)
- vérifier avec Greg qu'il comprend la nouvelle version de `train.py` avant les vrais runs

## Greg

Section remplie par Maksim les 7 et 9 novembre à partir des commits et des résultats poussés par Greg.

### PC GPU et première version de train.py (tâches 0.4 et 1.6), les 5 et 6 novembre

- **0.4** PC d'entraînement prêt : PyTorch avec CUDA sur la RTX 2000 Ada.
- **1.6** Première version de `train.py` (boucle d'entraînement, courbes, meilleur modèle selon le F1, checkpoints). Elle a ensuite été corrigée et simplifiée par Maksim le 7 novembre (voir sa section).

### Les 5 entraînements (tâches 3.1 et 3.2), le 7 novembre

- Lancés sur le dataset complet (10 636 images de train, 2 660 de validation), avec les réglages par défaut de `train.py` : SGD momentum 0.9, batch 64, augmentation activée, graine 42.
- Résultats dans `results/<config>/` :

| Configuration        | Époques | lr   | F1 macro val | Paramètres | Frugalité | Durée    |
| -------------------- | ------: | ---- | -----------: | ---------: | --------: | -------: |
| baseline             |      20 | 1e-4 |        0,335 |     1,08 M |     0,179 |  6,6 min |
| MobileNetV2 frozen   |      10 | 1e-2 |        0,874 |     2,24 M |     0,667 |  4,2 min |
| ResNet-18 frozen     |      10 | 1e-2 |        0,859 |    11,18 M |     0,584 |  3,8 min |
| MobileNetV2 finetune |      15 | 1e-3 |        0,925 |     2,24 M |     0,718 | 11,6 min |
| ResNet-18 finetune   |      15 | 1e-3 |        0,913 |    11,18 M |     0,638 |  7,9 min |

- Meilleure configuration : **MobileNetV2 en fine-tuning**, à la fois sur le F1 et sur la frugalité. ResNet-18 a 5 fois plus de paramètres et fait moins bien dans les deux modes.
- Premières observations pour le diagnostic biais / variance (tâche 4.2) :
  - baseline : sous-apprentissage (accuracy train 0,38, validation 0,36 après 20 époques). Lancée sur 20 époques au lieu de 10 parce qu'elle apprend lentement, à noter dans le protocole d'entraînement du rapport (6.3) ;
  - modes gelés : plafond vers 0,85-0,87 dès l'époque 3, accuracy train un peu sous la validation (effet de l'augmentation), pas de surapprentissage ;
  - fine-tuning : début de variance. Pour MobileNetV2, accuracy train 0,978 contre 0,915 en validation, loss de validation minimale à l'époque 11 (0,290) puis légère remontée (0,308), alors que le F1 monte encore jusqu'à l'époque 15.
- Classes les plus faibles du meilleur modèle : `dessert` (0,86), `bread` et `fried` (0,88). Confusions principales : dessert → bread (12), fried → bread (11), dessert → meat (11), fried → meat (10). Les petites classes sont au contraire les mieux reconnues (`rice` 0,99, `noodles-pasta` 0,98).

### Run sans augmentation (tâche 3.4), le 8 novembre

- `python train.py --model mobilenetv2 --mode finetune --epochs 15 --no-augment`, résultats dans `results/mobilenetv2_finetune_noaug/`.
- Mêmes réglages que le run avec augmentation de la 3.2 (vérifié dans les deux `config.json`, seul `no_augment` change), comme prévu dans `PROTOCOLE_AUGMENTATION.md`.

| MobileNetV2 finetune             | Avec augmentation | Sans augmentation |
| -------------------------------- | ----------------: | ----------------: |
| Meilleur F1 macro val (époque)   |        0,925 (15) |        0,911 (11) |
| Accuracy train, dernière époque  |            97,8 % |           99,96 % |
| Accuracy val, dernière époque    |            91,5 % |            89,9 % |
| Loss val minimale (époque)       |        0,290 (11) |         0,341 (6) |
| Loss val, dernière époque        |             0,308 |             0,366 |
| Frugalité                        |             0,718 |             0,703 |

- Sans augmentation, le train est appris presque par cœur (99,6 % dès l'époque 9) et la loss de validation ne baisse plus après l'époque 6, alors qu'avec augmentation le F1 monte encore à l'époque 15.
- Par classe, la plus forte baisse est sur `dairy` (0,912 → 0,858) puis `egg` (0,920 → 0,882). `rice` (0,987) et `noodles-pasta` (0,983 → 0,974) bougent peu.
- L'analyse détaillée est à faire dans la tâche 4.4.

### Prochaines tâches

- **6.3** et **6.5** parties du rapport (protocole d'entraînement, biais / variance)

### Points à régler

- Mettre à jour cette section soi-même aux prochaines séances (consigne du README), et ajouter ses lignes dans le tableau « Usage de l'IA » (pour la première version de `train.py` notamment).
- Le `results/baseline/config.json` affichait `"mode": "frozen"` (valeur par défaut de `--mode`, sans effet sur la baseline). Corrigé par Maksim : `train.py` écrit maintenant `"aucun (entraînée de zéro)"` pour la baseline, et le fichier existant a été modifié de la même façon (seul ce champ change, pas les résultats).

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

### Confusion et analyse des erreurs (tâche 4.3), le 9 novembre

Script : `analyse_erreurs.py`, calculé uniquement depuis `results/mobilenetv2_finetune/val_predictions.csv`
(modèle final choisi en 4.1). Figures : `results/figures/confusion_mobilenetv2_finetune.png`,
`f1_par_classe_mobilenetv2_finetune.png` et `erreurs_mobilenetv2_finetune.png`.

**225 erreurs sur 2 660 images de validation (8,5 %).**

| Classe | Précision | Rappel | F1 | Images |
|---|---|---|---|---|
| dessert | 0,881 | 0,848 | 0,864 | 400 |
| bread | 0,851 | 0,908 | 0,879 | 271 |
| fried | 0,910 | 0,855 | 0,882 | 235 |
| dairy | 0,920 | 0,904 | 0,912 | 115 |
| meat | 0,899 | 0,932 | 0,916 | 355 |
| egg | 0,920 | 0,920 | 0,920 | 263 |
| seafood | 0,952 | 0,896 | 0,923 | 241 |
| soup | 0,939 | 0,955 | 0,947 | 400 |
| vegetable-fruit | 0,949 | 0,984 | 0,966 | 188 |
| noodles-pasta | 0,983 | 0,983 | 0,983 | 117 |
| rice | 0,987 | 0,987 | 0,987 | 75 |

**Les erreurs se concentrent sur quatre classes : dessert, bread, fried et meat.** Ce sont les F1 les plus
bas, et les confusions les plus fréquentes se font entre elles : dessert → bread (12), fried → bread (11),
dessert → meat (11), fried → meat (10). Ces classes partagent des couleurs et des textures (doré, brun,
grillé) et dessert est la classe la plus variée (gâteaux, glaces, crèmes, tartes). On retrouve les paires
repérées dans `DATA.md` (fried / meat, bread / meat, dairy / dessert).

**Le déséquilibre n'explique pas les erreurs.** rice, la plus petite classe (75 images en validation, 376 au
total), a le meilleur F1 (0,987), et noodles-pasta suit (0,983). Ces classes ont un aspect très typique. À
l'inverse, dessert et bread font partie des classes les plus fournies et sont les moins bien reconnues.

**bread sert de classe « refuge ».** Sa précision (0,851) est nettement plus basse que son rappel (0,908) : le
modèle répond bread trop souvent, en particulier pour des plats dorés ou grillés.

**Le modèle est souvent trop sûr de lui.** La confiance moyenne est de 0,96 sur les bonnes réponses et de 0,77
sur les erreurs, mais 85 des 225 erreurs ont une confiance supérieure à 90 %. La confiance aide à repérer les
cas douteux, sans être fiable pour autant.

**Six erreurs commentées** (`erreurs_mobilenetv2_finetune.png`) :

| Image | Vraie → prédite (confiance) | Commentaire |
|---|---|---|
| meat/579.jpg | meat → bread (100 %) | Sandwich au pain grillé : le pain occupe presque toute l'image et la viande n'apparaît qu'en fine couche. Plusieurs aliments, la prédiction est défendable. |
| fried/705.jpg | fried → meat (100 %) | Poisson pané avec des frites : c'est à la fois du frit et du poisson. Les classes du dataset se recouvrent (mode de cuisson contre type d'aliment). |
| dairy/538.jpg | dairy → dessert (100 %) | Crème fouettée en pot, présentée comme un dessert. Classes proches : un produit laitier sucré ressemble à un dessert. |
| fried/225.jpg | fried → dessert (100 %) | Churro trempé dans du chocolat : c'est un aliment frit, mais aussi un dessert. L'étiquette est discutable, la prédiction aussi juste que la vérité. |
| dessert/532.jpg | dessert → seafood (100 %) | Bacs de glace au chocolat dans une vitrine en inox. Vraie erreur du modèle, sur une vue inhabituelle (vitrine, pas d'assiette), avec une confiance maximale. |
| bread/465.jpg | bread → dessert (35 %) | Burger, frites et coleslaw : plusieurs aliments et une étiquette peu évidente (pain, viande ou frit ?). Ici le modèle hésite, ce qui est le comportement attendu. |

**Hypothèses sur la vraie erreur (dessert → seafood).** Le modèle fait 6 fois cette erreur (dessert/82, 216,
318, 532, 1062, 1092). En regardant les six images :
- trois ont une feuille de menthe en décoration (82, 216, 1092), comme les herbes qu'on met souvent sur le
  poisson ; la tarte aux pommes (318) a des tranches orange et brillantes qui ressemblent à du saumon ou à des
  crevettes ;
- le modèle s'appuierait donc sur des indices de présentation (décoration verte, couleur orange, sauce
  nappée) plutôt que sur l'aliment lui-même ;
- pour dessert/532, la vue est inhabituelle : des bacs en inox dans une vitrine, sans assiette. Les desserts
  du train sont surtout photographiés à l'assiette, le modèle a probablement vu peu d'images de ce type.
  Le piment vert posé sur la glace peut aussi jouer.

Ce sont des hypothèses faites à l'œil, non vérifiées : il faudrait compter les images de vitrine dans le
train ou afficher les zones de l'image qui font décider le modèle (Grad-CAM) pour les confirmer. Le niveau de
confiance va dans le même sens : 5 des 6 erreurs sont hésitantes (40 à 82 %), seule la vitrine est à 100 %.

**Conclusion.** Sur ces six erreurs, une seule est une vraie erreur de reconnaissance (dessert/532). Les cinq
autres viennent du dataset lui-même : une seule étiquette par image alors que l'assiette contient plusieurs
aliments, et des catégories qui se recouvrent (fried est un mode de cuisson, les autres sont des types
d'aliments). Une partie des 8,5 % d'erreurs n'est donc pas corrigeable par un meilleur réseau. Ces six images
ont été choisies parmi les erreurs les plus sûres et les plus hésitantes : elles illustrent les types d'erreurs,
sans donner leur proportion exacte sur les 225.

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
| Maksim | Claude, Claude Code | Script d'analyse des runs et rédaction du diagnostic biais / variance à partir des courbes | analyse_entrainement.py, section 4.2 |
| Maksim | Claude Code         | Comparaison des runs avec / sans augmentation, mise à jour du suivi de Greg et du protocole              | suivi 3.4, protocole 3.3    |
| Maksim | Claude, Claude Code | Script de frugalité, figure d'iso-frugalité et rédaction de l'analyse | frugalite.py, section 4.1 |
| Oggy   | Claude Code         | Script d'analyse des erreurs, choix des images mal classées et rédaction des commentaires | analyse_erreurs.py, section 4.3 |
