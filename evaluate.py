"""
evaluate.py : métriques d'évaluation.
Responsable : Oggy (tâche 1.7)

À implémenter :
- prédictions d'un modèle sur un loader
- F1 macro, précision et rappel par classe, matrice de confusion (scikit-learn)
- score de frugalité : Frug = P - w / (1 + 1/R), w = 0.3 (unité de R à confirmer par le prof)
- export metrics.json et val_predictions.csv (image, vraie classe, classe prédite, confiance)
"""
