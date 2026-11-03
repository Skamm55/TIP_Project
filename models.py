"""
models.py : définition des réseaux.
Responsable : Julien (tâche 1.5)

À implémenter :
- get_model(name, mode, num_classes=11)
    name : "baseline" (CNN du TD), "mobilenetv2", "resnet18"
    mode : "frozen" (seule la tête est entraînée) ou "finetune"
- count_params(model)
- test sans dataset : torch.randn(2, 3, 224, 224) -> sortie de taille (2, 11)
"""
