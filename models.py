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
"""models.py : définition des modèles du projet Food-11.

Trois architectures sont disponibles :
  - baseline    : le petit CNN du TD, entraîné de zéro ;
  - mobilenetv2 : MobileNetV2 pré-entraîné sur ImageNet ;
  - resnet18    : ResNet-18 pré-entraîné sur ImageNet.

Pour les deux réseaux pré-entraînés, la dernière couche est remplacée par une
couche linéaire à 11 sorties (les 11 classes de Food-11), et deux modes existent :
  - frozen   : tout le réseau est gelé sauf la nouvelle tête (extracteur de features) ;
  - finetune : tout le réseau est entraîné (fine-tuning complet).

Les sorties sont des logits (pas de softmax) : la softmax est incluse dans la loss
d'entropie croisée (nn.CrossEntropyLoss).
"""

import torch
import torch.nn as nn
from torchvision import models as tv_models

NUM_CLASSES = 11
MODEL_NAMES = ("baseline", "mobilenetv2", "resnet18")
MODES = ("frozen", "finetune")


class BaselineCNN(nn.Module):
    """CNN du TD : conv 3x3 (8 filtres) -> BatchNorm -> ReLU -> MaxPool 2 -> FC.

    Pour une entrée 224x224, la convolution (sans padding) donne 222x222,
    puis le max-pooling donne 111x111 : la couche FC reçoit donc
    8 * 111 * 111 valeurs.
    """

    def __init__(self, num_classes=NUM_CLASSES, img_size=224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),  # 3 canaux RGB -> 8 filtres
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        # Taille de la carte de features après conv (sans padding) et pooling.
        side = (img_size - 2) // 2
        self.classifier = nn.Linear(8 * side * side, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # (batch, 8*side*side)
        return self.classifier(x)  # logits, sans softmax


def _build_mobilenetv2(num_classes, pretrained):
    """MobileNetV2 avec une nouvelle tête à num_classes sorties."""
    weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)
    # classifier = [Dropout, Linear(1280 -> 1000)] : on remplace seulement la Linear.
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _build_resnet18(num_classes, pretrained):
    """ResNet-18 avec une nouvelle tête à num_classes sorties."""
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _head_parameters(model, name):
    """Retourne les paramètres de la tête (la couche de classification)."""
    if name == "mobilenetv2":
        return model.classifier[1].parameters()
    if name == "resnet18":
        return model.fc.parameters()
    return model.classifier.parameters()  # baseline


def get_model(name, mode="finetune", num_classes=NUM_CLASSES, pretrained=True, img_size=224):
    """Construit et retourne le modèle demandé.

    Args:
        name: "baseline", "mobilenetv2" ou "resnet18".
        mode: "frozen" (seule la tête est entraînée) ou "finetune" (tout est entraîné).
              Ignoré pour la baseline, qui est toujours entièrement entraînée.
        num_classes: nombre de classes de sortie (11 pour Food-11).
        pretrained: charge les poids ImageNet pour mobilenetv2 et resnet18.
                    Mettre False pour les tests sans réseau ni téléchargement.
        img_size: taille de l'image d'entrée (utile seulement pour la baseline).
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Modèle inconnu : {name!r}. Choix possibles : {MODEL_NAMES}")
    if mode not in MODES:
        raise ValueError(f"Mode inconnu : {mode!r}. Choix possibles : {MODES}")

    if name == "baseline":
        return BaselineCNN(num_classes=num_classes, img_size=img_size)

    if name == "mobilenetv2":
        model = _build_mobilenetv2(num_classes, pretrained)
    else:
        model = _build_resnet18(num_classes, pretrained)

    if mode == "frozen":
        # On gèle tout le réseau...
        for p in model.parameters():
            p.requires_grad = False
        # ...puis on réactive uniquement la nouvelle tête.
        for p in _head_parameters(model, name):
            p.requires_grad = True

    return model


def count_params(model, trainable_only=False):
    """Nombre de paramètres du modèle (tous, ou seulement les entraînables)."""
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


def trainable_parameters(model):
    """Paramètres à passer à l'optimiseur (ceux dont requires_grad est True)."""
    return [p for p in model.parameters() if p.requires_grad]


if __name__ == "__main__":
    # Test sans dataset : une entrée aléatoire doit donner une sortie (2, 11).
    # pretrained=False pour ne rien télécharger ; le nombre de paramètres est le même.
    x = torch.randn(2, 3, 224, 224)
    for model_name in MODEL_NAMES:
        for model_mode in MODES:
            m = get_model(model_name, model_mode, pretrained=False)
            m.eval()
            with torch.no_grad():
                out = m(x)
            assert out.shape == (2, NUM_CLASSES), f"Mauvaise forme : {tuple(out.shape)}"
            print(
                f"{model_name:12s} {model_mode:9s} sortie {tuple(out.shape)}  "
                f"total {count_params(m):>10,d}  entraînables {count_params(m, True):>10,d}"
            )