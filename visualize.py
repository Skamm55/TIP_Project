"""
visualize.py : visualisation des feature maps de la 1re couche de convolution.
Responsable : Maksim (tâche 1.9)

Équivalent Python de Visualize_features.m du TD : on regarde ce que "voit" la
1re couche de convolution d'un réseau pré-entraîné sur une vraie image du dataset.

Exemples :
    python visualize.py                                   # feature maps MobileNetV2 sur une image de bread
    python visualize.py --model resnet18 --class soup --index 0
    python visualize.py --classes bread dessert meat seafood soup rice
    python visualize.py --model resnet18 --filters
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import models as tv_models

from data import (CLASSES, DEFAULT_DATA_DIR, FIGURES_DIR, IMAGENET_MEAN, IMAGENET_STD,
                  chemin_donnees, get_transforms, lister_images, pil_loader)

# taille des poids ImageNet, juste pour prévenir au 1er lancement
TAILLE_POIDS = {"mobilenetv2": "14 Mo", "resnet18": "45 Mo"}


def load_model(name: str, checkpoint: str | None = None) -> torch.nn.Module:
    """Charge MobileNetV2 ou ResNet-18 avec les poids ImageNet, ou un modèle entraîné si checkpoint est donné."""
    if checkpoint is not None:
        if not Path(checkpoint).exists():
            raise RuntimeError(f"Le fichier {checkpoint} n'existe pas.")
        try:
            from models import get_model
        except ImportError:
            raise RuntimeError("Impossible d'importer get_model depuis models.py "
                               "(models.py n'est peut-être pas encore écrit).")
        try:
            model = get_model(name, "finetune")
            poids = torch.load(checkpoint, map_location="cpu")
            # même format que dans predict.py : state_dict seul ou sous la clé "model"
            if "model" in poids:
                poids = poids["model"]
            model.load_state_dict(poids)
        except Exception as e:
            raise RuntimeError(f"Le chargement de {checkpoint} a échoué : {e}")
        print(f"Modèle {name} chargé depuis {checkpoint}")
    else:
        print(f"Chargement de {name} pré-entraîné sur ImageNet "
              f"(au 1er lancement les poids sont téléchargés, environ {TAILLE_POIDS[name]})")
        if name == "mobilenetv2":
            model = tv_models.mobilenet_v2(weights="DEFAULT")
        else:
            model = tv_models.resnet18(weights="DEFAULT")

    model.eval()
    return model


def first_layer(model: torch.nn.Module, name: str, stage: str = "act") -> torch.nn.Module:
    """Renvoie la couche à visualiser : la 1re conv seule (stage="conv") ou après l'activation (stage="act")."""
    if name == "mobilenetv2":
        # features[0] = conv 3x3 + BatchNorm + ReLU6
        if stage == "conv":
            return model.features[0][0]
        return model.features[0]
    else:
        # dans ResNet, model.relu n'est utilisé qu'une fois, juste après conv1 + bn1
        if stage == "conv":
            return model.conv1
        return model.relu


def nom_couche(name: str, stage: str) -> str:
    """Nom de la couche pour les titres des figures."""
    if name == "mobilenetv2":
        return "features[0][0] (conv)" if stage == "conv" else "features[0] (après ReLU6)"
    return "conv1 (conv)" if stage == "conv" else "conv1 + bn1 + relu"


def get_activations(model: torch.nn.Module, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Fait passer x dans le modèle et récupère la sortie de layer avec un forward hook.

    x est de forme (1, 3, H, W), la sortie de forme (C, H', W')."""
    sorties = []

    def hook(module, entree, sortie):
        sorties.append(sortie.detach())

    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    h.remove()  # sinon le hook resterait accroché au modèle
    return sorties[0][0]


def choisir_image(image: str | None, classe: str | None, index: int | None, seed: int) -> tuple[Path, str]:
    """Renvoie (chemin de l'image, classe) selon les options --image, --class et --index."""
    if image is not None:
        chemin = Path(image)
        if not chemin.exists():
            raise RuntimeError(f"L'image {image} n'existe pas.")
        # si l'image vient du train, le dossier parent donne la classe
        classe = chemin.parent.name if chemin.parent.name in CLASSES else "image"
        return chemin, classe

    if classe is None:
        classe = "bread"
    if classe not in CLASSES:
        raise RuntimeError(f"Classe '{classe}' inconnue, les classes possibles sont : {', '.join(CLASSES)}")

    images = sorted(lister_images(chemin_donnees(DEFAULT_DATA_DIR) / "train" / classe))
    if index is None:
        random.seed(seed)
        return random.choice(images), classe
    if index < 0 or index >= len(images):
        raise RuntimeError(f"Index {index} invalide, la classe {classe} a {len(images)} images.")
    return images[index], classe


def preparer(chemin: Path) -> torch.Tensor:
    """Charge une image et applique les transformations de validation, renvoie un tenseur (1, 3, 224, 224)."""
    return get_transforms(train=False)(pil_loader(chemin)).unsqueeze(0)


def denormaliser(x: torch.Tensor) -> torch.Tensor:
    """Annule la normalisation ImageNet pour afficher l'image avec ses vraies couleurs, renvoie (H, W, 3)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = x[0] * std + mean
    return img.clamp(0, 1).permute(1, 2, 0)


def canaux_a_afficher(acts: torch.Tensor, n: int, select: str = "top") -> list[int]:
    """Renvoie les n canaux les plus actifs (activation moyenne) ou les n premiers."""
    if select == "first":
        return list(range(n))
    moyennes = acts.mean(dim=(1, 2))
    return moyennes.argsort(descending=True)[:n].tolist()


def plot_feature_maps(x: torch.Tensor, acts: torch.Tensor, titre: str, chemin: Path, select: str = "top") -> None:
    """Grille 4x4 : l'image d'entrée puis 15 canaux de la couche en niveaux de gris."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.5))
    axes = axes.flatten()

    axes[0].imshow(denormaliser(x))
    axes[0].set_title("Image d'entrée")

    canaux = canaux_a_afficher(acts, 15, select)
    for i, k in enumerate(canaux):
        # imshow fait un min-max par défaut, donc chaque canal est normalisé tout seul
        axes[i + 1].imshow(acts[k], cmap="gray")
        axes[i + 1].set_title(f"Canal {k}")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(titre)
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()


def plot_classes(model: torch.nn.Module, layer: torch.nn.Module, classes: list[str], titre: str,
                 chemin: Path, seed: int = 42) -> None:
    """Une ligne par classe : une image au hasard puis ses 7 canaux les plus actifs."""
    random.seed(seed)
    fig, axes = plt.subplots(len(classes), 8, figsize=(16, 2.2 * len(classes)))
    if len(classes) == 1:
        axes = [axes]  # sinon axes n'est pas un tableau 2D

    for i, classe in enumerate(classes):
        images = sorted(lister_images(chemin_donnees(DEFAULT_DATA_DIR) / "train" / classe))
        chemin_img = random.choice(images)
        x = preparer(chemin_img)
        acts = get_activations(model, layer, x)

        axes[i][0].imshow(denormaliser(x))
        axes[i][0].set_title(f"{classe} ({chemin_img.stem})")
        for j, k in enumerate(canaux_a_afficher(acts, 7)):
            axes[i][j + 1].imshow(acts[k], cmap="gray")
            axes[i][j + 1].set_title(f"Canal {k}")
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(titre)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # laisse de la place au titre en haut
    plt.savefig(chemin, dpi=150)
    plt.close()


def plot_filters(conv: torch.nn.Conv2d, titre: str, chemin: Path) -> None:
    """Affiche les noyaux de la 1re convolution comme des petites images RGB (8 colonnes)."""
    poids = conv.weight.detach()  # forme (C_out, 3, k, k)
    n = poids.shape[0]
    n_lignes = (n + 7) // 8

    fig, axes = plt.subplots(n_lignes, 8, figsize=(8 * 1.2, n_lignes * 1.3))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            # chaque filtre est ramené entre 0 et 1 pour pouvoir l'afficher
            f = poids[i]
            f = (f - f.min()) / (f.max() - f.min())
            axes[i].imshow(f.permute(1, 2, 0), interpolation="nearest")
            axes[i].set_title(str(i), fontsize=8)

    fig.suptitle(titre)
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mobilenetv2", choices=["mobilenetv2", "resnet18"])
    parser.add_argument("--stage", default="act", choices=["act", "conv"])
    parser.add_argument("--checkpoint", default=None, help="modèle entraîné (.pth), optionnel")
    parser.add_argument("--image", default=None, help="chemin d'une image précise")
    parser.add_argument("--class", dest="classe", default=None, help="classe de l'image")
    parser.add_argument("--index", type=int, default=None, help="numéro de l'image dans la classe")
    parser.add_argument("--select", default="top", choices=["top", "first"])
    parser.add_argument("--classes", nargs="+", default=None, help="comparaison entre plusieurs classes")
    parser.add_argument("--filters", action="store_true", help="affiche les filtres de la 1re couche")
    parser.add_argument("--maps", action="store_true", help="feature maps d'une image (par défaut)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # sans option, on fait les feature maps d'une image
    if not args.classes and not args.filters:
        args.maps = True

    try:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        model = load_model(args.model, args.checkpoint)
        layer = first_layer(model, args.model, args.stage)
        couche = nom_couche(args.model, args.stage)

        if args.maps:
            chemin_img, classe = choisir_image(args.image, args.classe, args.index, args.seed)
            x = preparer(chemin_img)
            acts = get_activations(model, layer, x)
            print(f"Image : {chemin_img.name} ({classe})")
            print(f"Sortie de {couche} : {acts.shape[0]} canaux de taille {acts.shape[1]}x{acts.shape[2]}")

            titre = f"{args.model} - {couche} - {classe} - {chemin_img.name}"
            # suffixe _conv pour ne pas écraser la figure après activation
            suffixe = "_conv" if args.stage == "conv" else ""
            sortie = FIGURES_DIR / f"featuremaps_{args.model}_{classe}_{chemin_img.stem}{suffixe}.png"
            plot_feature_maps(x, acts, titre, sortie, args.select)
            print("Figure enregistrée :", sortie)

        if args.classes:
            for c in args.classes:
                if c not in CLASSES:
                    raise RuntimeError(f"Classe '{c}' inconnue, les classes possibles sont : {', '.join(CLASSES)}")
            suffixe = "_conv" if args.stage == "conv" else ""
            sortie = FIGURES_DIR / f"featuremaps_{args.model}_classes{suffixe}.png"
            plot_classes(model, layer, args.classes, f"{args.model} - {couche}", sortie, args.seed)
            print("Figure enregistrée :", sortie)

        if args.filters:
            conv = first_layer(model, args.model, "conv")
            k = conv.kernel_size[0]
            print(f"Filtres de la 1re couche : {k}x{k}x{conv.out_channels} "
                  f"({conv.out_channels} noyaux {k}x{k} sur les {conv.in_channels} canaux RGB)")
            sortie = FIGURES_DIR / f"filters_{args.model}.png"
            plot_filters(conv, f"{args.model} : {conv.out_channels} filtres {k}x{k} de la 1re couche", sortie)
            print("Figure enregistrée :", sortie)

    except (RuntimeError, FileNotFoundError) as e:
        print("Erreur :", e)
        sys.exit(1)
