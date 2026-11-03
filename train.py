"""
train.py : entraînement d'une configuration.
Responsable : Greg (tâche 1.6)

Exemple d'usage visé :
    python train.py --model mobilenetv2 --mode frozen --epochs 10
    python train.py --model mobilenetv2 --mode frozen --subset 300 --epochs 1   (test en petit)

À implémenter :
- arguments : --model --mode --epochs --lr --batch-size --subset --img-size --no-augment --data-dir --out-dir
- device automatique (cuda si disponible, sinon cpu), précision mixte sur GPU
- checkpoint à chaque époque + reprise, sauvegarde du meilleur modèle (F1 de validation)
- sauvegarde des métriques par époque et des courbes dans results/
"""
