# Revue globale du projet (9 novembre 2026)

Revue faite après le commit de la 4.6. Rendu Moodle le 15 novembre à 23h59, présentations les 16 et 17.

## 1. État des tâches

État d'après la page de suivi (cases cochées) et l'historique Git.

| Phase                 | Fait (commit)                                                                                                                                         | Reste                                                         | Qui                         |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------- |
| 0 Lancement           | 0.1 à 0.5 (`3567363` structure du dépôt)                                                                                                              | rien                                                          |                             |
| 1 Données et code     | 1.2 `3749f41`, 1.3 `64ca502`, 1.5 `b2ed73b`, 1.6 `294eddf`, 1.7 `26ed304`, 1.8 `9435ee0`, 1.9 `4929b16`, 1.1 et 1.4 (sans commit dédié)              | 1.10 non cochée, 1.11 squelette du rapport et journal IA      | Julien (1.10), Maksim (1.11) |
| 2 Intégration         | 2.1 `827a16c`, `0e50bfa`, `acfb352` ; 2.2 `3c1860a` ; 2.3, 2.4, 2.5 cochées                                                                           | rien                                                          |                             |
| 3 Entraînements       | 3.1 et 3.2 `e41ef77` à `99a0b76`, 3.3 `3c1860a`, 3.4 `cb458dd`                                                                                         | rien                                                          |                             |
| 4 Analyse             | 4.0 (poids reçus), 4.1 `035cb6c`, 4.2 `3238ad6` `74c72fc`, 4.3 `2801574`, 4.4 `39cd2df`, 4.5 `8f8069d`, 4.6 `bba62e5` `5166d76`                        | cocher 4.6 sur la page (case de groupe)                       | groupe                      |
| 5 Test et soumission  | rien                                                                                                                                                  | 5.1 JSON, 5.2 essai serveur, 5.3 description, 5.4 soumission  | Maksim, Oggy, Julien        |
| 6 Rapport             | rien                                                                                                                                                  | 6.1 à 6.11                                                    | tous                        |
| 7 Présentation        | rien                                                                                                                                                  | 7.1 à 7.6                                                     | tous, pilotage Oggy         |
| 8 Rendu               | rien                                                                                                                                                  | 8.1 clone propre, 8.2 README final, 8.3 archive Moodle        | Julien, tous, Maksim        |

La 4.1 était attribuée à Julien sur la page de suivi mais a été faite par Maksim.

## 2. Inventaire des fichiers suivis

| Fichier                                 | Rôle                                                                                     |
| --------------------------------------- | ---------------------------------------------------------------------------------------- |
| `data.py`                               | chargement Food-11, split stratifié 80/20 (graine 42), transformations, stats du dataset |
| `models.py`                             | baseline du TD, MobileNetV2, ResNet-18, modes frozen / finetune, `count_params`          |
| `train.py`                              | entraînement (SGD momentum), courbes, meilleur modèle selon le F1, appel à `evaluer`     |
| `evaluate.py`                           | F1 macro, métriques par classe, matrice de confusion, frugalité, export des résultats    |
| `predict.py`                            | JSON des prédictions sur le test, avec vérification du format                            |
| `visualize.py`                          | feature maps et filtres de la 1re couche                                                 |
| `analyse_entrainement.py`               | figures des courbes (4.2), figures avec / sans augmentation (4.4), `synthese_runs.csv`    |
| `frugalite.py`                          | tableau de frugalité et figure F1 / paramètres (4.1)                                     |
| `analyse_erreurs.py`                    | matrice de confusion, scores par classe, planche d'erreurs (4.3)                         |
| `README.md`                             | installation et commandes                                                                |
| `DATA.md`                               | analyse exploratoire du dataset (1.2, 1.4)                                               |
| `PROTOCOLE_AUGMENTATION.md`             | protocole de l'étude d'augmentation (3.3)                                                |
| `DECISION_MODELE_FINAL.md`              | compte rendu du choix du modèle final (4.6)                                              |
| `SUIVI_AVANCEMENT.md`                   | journal par personne et tableau d'usage de l'IA                                          |
| `RAPPORT_REVUE.md`                      | ce rapport                                                                               |
| `requirements.txt`, `.gitignore`        | dépendances, fichiers exclus de Git                                                      |
| `notebooks/.gitkeep`, `results/.gitkeep` | fichiers vides qui gardent les dossiers dans Git                                        |
| `results/<run>/` (6 runs)               | `config.json`, `history.csv`, `metrics.json`, `val_predictions.csv`, `curves.png`        |
| `results/test_surapprentissage/`        | test de surapprentissage (2.2)                                                           |
| `results/figures/` (17 figures)         | figures du dataset, des courbes, de la frugalité, des erreurs et des feature maps         |
| `results/synthese_runs.csv`             | chiffres clés de chaque run (4.2)                                                        |
| `results/frugalite.csv`                 | tableau de frugalité (4.1)                                                               |
| `results/split_seed42.csv`              | détail du split train / validation, cité dans `DATA.md`                                  |
| `results/predictions.json`              | essai de JSON du 9 novembre, fait avant la décision finale                               |

## 3. Résultats clés

| Run                                 | F1 validation | Paramètres | Frugalité |
| ----------------------------------- | ------------: | ---------: | --------: |
| Baseline (TD)                       |        0,3346 |     1,08 M |     0,178 |
| MobileNetV2 gelé                    |        0,8740 |     2,24 M |     0,667 |
| ResNet-18 gelé                      |        0,8589 |    11,18 M |     0,583 |
| **MobileNetV2 fine-tuning**         |    **0,9252** | **2,24 M** | **0,718** |
| ResNet-18 fine-tuning               |        0,9134 |    11,18 M |     0,638 |
| MobileNetV2 fine-tuning sans augm.  |        0,9108 |     2,24 M |     0,703 |

Modèle retenu (4.6) : MobileNetV2 pré-entraîné sur ImageNet, fine-tuning complet, avec augmentation,
poids `results/mobilenetv2_finetune/best.pth` (époque 15, pas sur Git).

## 4. Cohérence entre les documents et les résultats

Aucun lien mort : tous les fichiers et figures cités dans les `.md` existent.

Écarts relevés, non corrigés :

| Où                                              | Écrit                                        | Valeur dans `results/`                                      |
| ----------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------- |
| `SUIVI_AVANCEMENT.md`, tableau de Greg (3.1/3.2) | frugalité baseline 0,179, ResNet-18 gelé 0,584 | 0,17849 et 0,58349, soit 0,178 et 0,583 (comme en 4.1)    |
| `SUIVI_AVANCEMENT.md`, 4.2 « modèles gelés »    | F1 qui plafonne entre 0,86 et 0,87           | ResNet-18 gelé ne dépasse pas 0,859                         |
| `SUIVI_AVANCEMENT.md`, 4.4 « effet par classe » | rice et noodles-pasta entre 0,98 et 0,99     | noodles-pasta sans augmentation : 0,974                     |
| `DECISION_MODELE_FINAL.md`, étape 5             | +0,005 sur les 4 dernières époques           | +0,007 de l'époque 11 à 15, +0,005 de l'époque 12 à 15      |
| `DECISION_MODELE_FINAL.md`, vigilance           | regénérer le JSON « avec le bon fichier de poids » | l'essai utilisait déjà `results/mobilenetv2_finetune/best.pth` |
| `DATA.md`, classes proches                      | « à vérifier avec la matrice de confusion, tâche 4.4 » | la matrice de confusion est la tâche 4.3            |

Sections dépassées ou incomplètes :
- `SUIVI_AVANCEMENT.md`, prochaines tâches de Maksim : « vérifier avec Greg qu'il comprend la nouvelle
  version de `train.py` avant les vrais runs » n'a plus d'objet (tous les runs sont faits).
- `SUIVI_AVANCEMENT.md`, section de Julien : « Rien de consigné », alors que `models.py` est de lui.
- `README.md` : le tableau des fichiers ne liste ni `analyse_entrainement.py`, ni `frugalite.py`, ni
  `analyse_erreurs.py`, ni `DECISION_MODELE_FINAL.md` ; l'exemple de `predict.py` utilise le modèle gelé
  au lieu du modèle final. À reprendre en 8.2.
- `README.md` annonce « un notebook par personne », mais `notebooks/` est vide.

## 5. Modifications faites pendant cette revue

Commit `0997e31` « revue : nettoyage du code et des commentaires ». Aucun changement de comportement,
aucune fonction publique renommée.

| Fichier              | Modification                                                                                                     |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `predict.py`         | exemple d'usage corrigé (`checkpoints/best.pth` n'existe pas) ; import de `get_model` remonté en haut du fichier, le commentaire qui le justifiait (« pour que --self-test marche sans models.py ») n'avait plus d'objet |
| `visualize.py`       | suppression du `try / except ImportError` sur `models.py` (« pas encore écrit »), devenu inutile ; import en haut du fichier |
| `models.py`          | docstring de `get_model` sans section « Args » ; commentaires qui paraphrasaient le gel des couches remplacés par un seul qui dit pourquoi ; en-tête au même format que les autres fichiers |
| `data.py`            | docstrings de `__len__` et `__getitem__` retirées (elles répétaient le code)                                     |
| `analyse_erreurs.py` | docstring ajoutée à `plot_confusion` et `plot_f1_par_classe`, commentaire sur `matplotlib.use("Agg")` comme ailleurs |
| `frugalite.py`       | en-tête « Responsable : Maksim (tâche 4.1) » comme les autres fichiers                                           |

Pas d'import inutilisé (vérifié avec pyflakes), pas de fichier temporaire suivi par Git.

Tests après modification :

| Test                                                                                            | Résultat                                                         |
| ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `python data.py --subset 300`                                                                   | OK : 13 296 images, 239 / 60 après split, batch (32, 3, 224, 224), 3 097 images de test |
| `python train.py --model mobilenetv2 --mode frozen --subset 300 --epochs 1 --out-dir results/_test_revue` | OK : 1 époque, évaluation et 6 fichiers écrits ; dossier supprimé ensuite |
| `python predict.py --self-test`                                                                 | OK : 4 tests                                                     |
| `python models.py`, `python evaluate.py`                                                        | OK : 6 configurations, 5 tests                                   |
| `python visualize.py`                                                                           | OK : `featuremaps_mobilenetv2_bread_957.png`                     |
| `python analyse_entrainement.py`, `python frugalite.py`, `python analyse_erreurs.py`            | OK : `git diff --stat results/` vide, CSV et figures identiques  |

## 6. Suppressions proposées

Toutes les recommandations ci-dessous ont été validées par Maksim et appliquées dans le commit
« revue : suppressions validées » : `predictions.json`, `results/.gitkeep` et `notebooks/.gitkeep`
retirés, `trainable_parameters` supprimée, `evaluate.py` utilise `count_params` de `models.py`, ligne
`checkpoints/` retirée du `.gitignore`, règle « un notebook par personne » retirée du README. Les trois
figures non citées sont gardées. Tests refaits ensuite : tous OK, `results/` inchangé.

| Élément                                         | Recommandation                     | Raison                                                                                       |
| ----------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------- |
| `results/predictions.json`                      | retirer de Git (`git rm`)          | essai fait avant la décision finale, commité par erreur dans `a38110b` « 4.1 » ; regénéré en 5.1 |
| `trainable_parameters` (`models.py`)            | supprimer                          | jamais appelée : `train.py` construit lui-même la liste des paramètres à entraîner           |
| `compter_params` (`evaluate.py`)                | remplacer par `count_params` de `models.py` | doublon exact ; à faire avec l'accord d'Oggy, `evaluate.py` importerait alors `models.py` |
| ligne `checkpoints/` du `.gitignore`            | supprimer                          | plus aucun script n'écrit dans ce dossier (`*.pth` suffit)                                   |
| reste du `.gitignore` (modèle GitHub générique) | garder                             | inoffensif                                                                                   |
| `results/.gitkeep`                              | supprimer                          | le dossier contient d'autres fichiers                                                        |
| `notebooks/.gitkeep` et `notebooks/`            | supprimer, ou y mettre les notebooks | dossier vide ; sinon retirer la règle du README                                            |
| `featuremaps_mobilenetv2_bread_957.png`, `featuremaps_mobilenetv2_bread_957_conv.png`, `filters_mobilenetv2.png` | garder ou supprimer selon le rapport | cités dans aucun `.md` ; utiles si la partie 6.7 compare la sortie conv / après ReLU6, ou les filtres 3×3 de MobileNetV2 aux 7×7 de ResNet-18 |

## 7. Risques et points ouverts avant le rendu

- **Calendrier** : à 6 jours du rendu, les phases 5 à 8 n'ont aucune tâche cochée (rapport 0 / 11,
  slides 0 / 6).
- **JSON final (5.1)** : à regénérer avec `results/mobilenetv2_finetune/best.pth`. Les poids ne sont que
  sur le PC de Maksim et celui de Greg.
- **`seafood`** : 2,2 % des prédictions sur le test contre 8,5 % sur la validation. À regarder avec le F1
  par classe du serveur (5.2), avant la soumission unique (5.4).
- **Unité de R** : toujours pas confirmée par le prof. Le choix du modèle n'en dépend pas, mais le score de
  frugalité annoncé dans le rapport, si.
- **Rapport et slides** : le squelette (1.11) n'existe pas encore ; 10 pages maximum.
- **Annexe IA (6.10)** : le tableau de `SUIVI_AVANCEMENT.md` est rempli surtout par Maksim ; Greg et Julien
  n'ont pas ajouté leurs lignes.
- **Archive sans fichiers binaires** : pas de `.pth` ni de données sur Git. En revanche `results/` contient
  des PNG ; à confirmer avec le prof si « without binary files » concerne aussi les figures.
