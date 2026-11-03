"""
data.py : chargement du dataset Food-11, split train / validation, transformations.
Responsable : Maksim (tâches 1.2, 1.3, 1.4)

À implémenter :
- chargement du train depuis data/food-11/ (un dossier par classe) et du test sans étiquettes
- split 80/20 stratifié avec graine fixe (seed=42)
- transformations train (RandomResizedCrop 224, flip) et validation / test (Resize 256, CenterCrop 224)
- normalisation
- options : --subset N, --no-augment, num_workers réglable
"""
