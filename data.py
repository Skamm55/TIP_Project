"""
data.py : chargement du dataset Food-11, split train / validation, transformations.
Responsable : Maksim (tâches 1.2, 1.3, 1.4)

Contenu :
- chargement du train depuis data/food-11/ (un dossier par classe) et du test sans étiquettes
- split 80/20 stratifié avec graine fixe (seed=42)
- transformations train (RandomResizedCrop 224, flip) et validation / test (Resize 256, CenterCrop 224)
- normalisation
- options : --subset N, --no-augment, num_workers réglable
"""

import argparse
import csv
import random
import statistics
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CLASSES = ["bread", "dairy", "dessert", "egg", "fried", "meat",
           "noodles-pasta", "rice", "seafood", "soup", "vegetable-fruit"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_DATA_DIR = "data/food-11"

# dossier du dépôt (là où se trouve data.py), pour que les chemins marchent
# même si on lance le script depuis un autre dossier
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def chemin_donnees(data_dir: str) -> Path:
    """Transforme data_dir en chemin et vérifie que le dossier existe."""
    chemin = Path(data_dir)
    if not chemin.is_absolute():
        chemin = ROOT / chemin
    if not chemin.exists():
        raise FileNotFoundError(
            f"Le dossier {chemin} n'existe pas. Il faut dézipper l'archive Food-11 dans "
            f"le dossier data/ du projet pour avoir data/food-11/train et data/food-11/test."
        )
    return chemin


def lister_images(dossier: Path) -> list[Path]:
    """Renvoie la liste des images (.jpg, .jpeg, .png) d'un dossier."""
    images = []
    for p in dossier.iterdir():
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            images.append(p)
    return images


def pil_loader(path: str | Path, img_size: int = 224) -> Image.Image:
    """Ouvre une image en RGB. Pour les JPEG, décode directement à une taille réduite
    (au moins 2 × img_size de côté) grâce au mode draft de PIL : beaucoup plus rapide
    sur les très grandes images, sans perte utile puisqu'on redimensionne ensuite à img_size."""
    with Image.open(path) as img:
        # x2 pour garder assez de pixels pour RandomResizedCrop (qui peut garder que 50% de l'image)
        img.draft("RGB", (2 * img_size, 2 * img_size))
        return img.convert("RGB")


def get_transforms(train: bool, img_size: int = 224, augment: bool = True):
    """Transformations pour le train (avec augmentation) ou pour la validation / le test."""
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # 256 pour 224, on garde le même rapport si on change img_size
        return transforms.Compose([
            transforms.Resize(round(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class TestDataset(Dataset):
    """Dataset pour les images de test, qui n'ont pas d'étiquette.

    Renvoie (image, nom) avec nom = nom du fichier sans extension, ex : "0".
    """

    def __init__(self, root: str = DEFAULT_DATA_DIR + "/test", transform=None, img_size: int = 224):
        """root : dossier des images de test, transform : transformations à appliquer,
        img_size : taille finale des images (sert pour le décodage réduit des JPEG)."""
        self.root = chemin_donnees(root)
        if transform is None:
            transform = get_transforms(train=False, img_size=img_size)
        self.transform = transform
        self.img_size = img_size
        # tri par numéro et pas par ordre alphabétique (sinon 0, 1, 10, 100...)
        self.paths = sorted(lister_images(self.root), key=lambda p: int(p.stem))

    def __len__(self) -> int:
        """Nombre d'images de test."""
        return len(self.paths)

    def __getitem__(self, idx: int):
        """Renvoie l'image transformée et son nom."""
        path = self.paths[idx]
        img = pil_loader(path, self.img_size)
        return self.transform(img), path.stem


def sous_echantillon(targets: list[int], n: int, seed: int) -> list[int]:
    """Prend environ n indices en gardant les proportions des classes (au moins 2 par classe)."""
    random.seed(seed)
    indices = []
    for c in range(len(CLASSES)):
        indices_classe = [i for i in range(len(targets)) if targets[i] == c]
        # nombre d'images à prendre pour cette classe
        n_c = round(n * len(indices_classe) / len(targets))
        # il faut au moins 2 images par classe sinon le split stratifié plante
        if n_c < 2:
            n_c = 2
        indices += random.sample(indices_classe, n_c)
    return sorted(indices)


def sauver_split(dataset, train_idx: list[int], val_idx: list[int], data_dir: Path, seed: int) -> None:
    """Enregistre dans un csv quelle image est dans le train et laquelle dans la validation."""
    lignes = []
    for i in train_idx:
        path, label = dataset.samples[i]
        lignes.append([Path(path).relative_to(data_dir).as_posix(), CLASSES[label], "train"])
    for i in val_idx:
        path, label = dataset.samples[i]
        lignes.append([Path(path).relative_to(data_dir).as_posix(), CLASSES[label], "val"])
    lignes.sort()

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"split_seed{seed}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chemin", "classe", "ensemble"])
        writer.writerows(lignes)


def get_dataloaders(data_dir: str = DEFAULT_DATA_DIR, img_size: int = 224, batch_size: int = 32,
                    num_workers: int = 0, subset: int | None = None, augment: bool = True,
                    val_ratio: float = 0.2, seed: int = 42):
    """Crée les DataLoaders de train et de validation avec un split stratifié.

    Renvoie (train_loader, val_loader, class_names).
    """
    train_dir = chemin_donnees(data_dir) / "train"

    # On crée deux ImageFolder sur les mêmes images : un avec les transformations
    # du train et un avec celles de la validation. Si on utilisait le même objet
    # pour les deux, changer la transformation de l'un changerait aussi l'autre.
    # partial et pas lambda : sous Windows les workers doivent pouvoir "pickler" le loader
    loader = partial(pil_loader, img_size=img_size)
    train_full = datasets.ImageFolder(train_dir, transform=get_transforms(True, img_size, augment), loader=loader)
    val_full = datasets.ImageFolder(train_dir, transform=get_transforms(False, img_size), loader=loader)

    class_names = train_full.classes
    if class_names != CLASSES:
        raise ValueError(f"Classes trouvées : {class_names}, classes attendues : {CLASSES}")

    targets = train_full.targets
    if subset is not None:
        indices = sous_echantillon(targets, subset, seed)
    else:
        indices = list(range(len(targets)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=[targets[i] for i in indices],
        random_state=seed,
    )

    # pas de csv avec subset sinon on écrase le split complet
    if subset is None:
        sauver_split(train_full, train_idx, val_idx, train_dir.parent, seed)

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=(num_workers > 0))
    return train_loader, val_loader, class_names


def get_test_loader(data_dir: str = DEFAULT_DATA_DIR, img_size: int = 224, batch_size: int = 32,
                    num_workers: int = 0) -> DataLoader:
    """DataLoader sur les images de test, dans l'ordre (pas de shuffle)."""
    test_set = TestDataset(chemin_donnees(data_dir) / "test", get_transforms(False, img_size), img_size)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=(num_workers > 0))


def class_counts(data_dir: str = DEFAULT_DATA_DIR) -> dict[str, int]:
    """Nombre d'images par classe dans le train."""
    train_dir = chemin_donnees(data_dir) / "train"
    counts = {}
    for classe in CLASSES:
        counts[classe] = len(lister_images(train_dir / classe))
    return counts


def compute_mean_std(data_dir: str = DEFAULT_DATA_DIR, img_size: int = 224, max_images: int | None = None):
    """Moyenne et écart-type par canal RGB sur les images du train (valeurs entre 0 et 1).

    On fait la somme et la somme des carrés en une seule passe,
    puis var = E[x^2] - E[x]^2.
    """
    tf = transforms.Compose([
        transforms.Resize(round(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(chemin_donnees(data_dir) / "train", transform=tf,
                                   loader=partial(pil_loader, img_size=img_size))
    if max_images is not None and max_images < len(dataset):
        random.seed(0)
        dataset = Subset(dataset, random.sample(range(len(dataset)), max_images))

    loader = DataLoader(dataset, batch_size=64)
    somme = torch.zeros(3, dtype=torch.float64)
    somme_carres = torch.zeros(3, dtype=torch.float64)
    nb_pixels = 0
    for images, _ in loader:
        images = images.double()
        somme += images.sum(dim=[0, 2, 3])
        somme_carres += (images ** 2).sum(dim=[0, 2, 3])
        nb_pixels += images.shape[0] * images.shape[2] * images.shape[3]

    mean = somme / nb_pixels
    std = torch.sqrt(somme_carres / nb_pixels - mean ** 2)
    return mean.tolist(), std.tolist()


def plot_counts(counts: dict[str, int]) -> None:
    """Diagramme en barres du nombre d'images par classe, trié."""
    classes_triees = sorted(counts, key=counts.get)
    valeurs = [counts[c] for c in classes_triees]

    plt.figure(figsize=(8, 5))
    plt.barh(classes_triees, valeurs, color="tab:blue")
    for i in range(len(valeurs)):
        plt.text(valeurs[i] + 15, i, str(valeurs[i]), va="center")
    plt.xlabel("Nombre d'images")
    plt.title("Nombre d'images par classe (train)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_counts.png", dpi=150)
    plt.close()


def plot_samples(data_dir: Path, n: int = 6, seed: int = 42) -> None:
    """Grille avec n images au hasard pour chaque classe (une ligne par classe)."""
    random.seed(seed)
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(n * 1.8, len(CLASSES) * 1.6))
    for i, classe in enumerate(CLASSES):
        images = random.sample(sorted(lister_images(data_dir / "train" / classe)), n)
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(pil_loader(images[j]))
            ax.set_xticks([])
            ax.set_yticks([])
        # nom de la classe au début de la ligne
        axes[i, 0].set_ylabel(classe, rotation=0, ha="right", va="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "samples_per_class.png", dpi=100)
    plt.close()


def tailles_images(data_dir: Path) -> None:
    """Affiche des stats sur la taille des images du train."""
    largeurs = []
    hauteurs = []
    for classe in CLASSES:
        for p in lister_images(data_dir / "train" / classe):
            # Image.open lit juste l'en-tête, l'image n'est pas décodée ici
            w, h = Image.open(p).size
            largeurs.append(w)
            hauteurs.append(h)
    formats = set(zip(largeurs, hauteurs))
    print("Largeur : min", min(largeurs), "max", max(largeurs), "médiane", statistics.median(largeurs))
    print("Hauteur : min", min(hauteurs), "max", max(hauteurs), "médiane", statistics.median(hauteurs))
    print("Nombre de formats différents :", len(formats))


def benchmark(data_dir: Path, n: int = 300, seed: int = 42) -> None:
    """Compare le temps de chargement de n images avec et sans le mode draft."""
    toutes = []
    for classe in CLASSES:
        toutes += sorted(lister_images(data_dir / "train" / classe))
    random.seed(seed)
    images = random.sample(toutes, n)

    debut = time.perf_counter()
    for p in images:
        Image.open(p).convert("RGB")
    t_normal = time.perf_counter() - debut

    debut = time.perf_counter()
    for p in images:
        pil_loader(p)
    t_draft = time.perf_counter() - debut

    print(f"Image.open().convert('RGB') : {t_normal:.2f} s")
    print(f"pil_loader (mode draft)     : {t_draft:.2f} s")
    print(f"Accélération : x{t_normal / t_draft:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--plot-counts", action="store_true")
    parser.add_argument("--samples", action="store_true")
    parser.add_argument("--sizes", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    data_dir = chemin_donnees(args.data_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    counts = class_counts(args.data_dir)
    print("Images par classe :")
    for classe in counts:
        print(f"  {classe:16s} {counts[classe]}")
    print("Total :", sum(counts.values()))

    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        subset=args.subset, augment=not args.no_augment, seed=args.seed)

    print(f"\nTrain : {len(train_loader.dataset)} images, validation : {len(val_loader.dataset)} images")
    # comptage par classe dans chaque partie
    targets = train_loader.dataset.dataset.targets
    for c in range(len(class_names)):
        n_train = sum(1 for i in train_loader.dataset.indices if targets[i] == c)
        n_val = sum(1 for i in val_loader.dataset.indices if targets[i] == c)
        print(f"  {class_names[c]:16s} train {n_train:5d}   val {n_val:4d}")

    images, labels = next(iter(train_loader))
    print("\nForme d'un batch :", tuple(images.shape))

    test_set = TestDataset(data_dir / "test")
    print("\nNombre d'images de test :", len(test_set))
    print("5 premiers noms :", [test_set[i][1] for i in range(5)])

    if args.plot_counts:
        plot_counts(counts)
        print("\nFigure enregistrée : results/figures/class_counts.png")

    if args.samples:
        plot_samples(data_dir, seed=args.seed)
        print("Figure enregistrée : results/figures/samples_per_class.png")

    if args.sizes:
        print("\nTailles des images du train :")
        tailles_images(data_dir)

    if args.benchmark:
        print("\nTemps de chargement de 300 images du train :")
        benchmark(data_dir, seed=args.seed)

    if args.stats:
        print("\nCalcul de la moyenne et de l'écart-type...")
        mean, std = compute_mean_std(args.data_dir, max_images=args.subset)
        print("Moyenne    :", [round(m, 3) for m in mean], " ImageNet :", list(IMAGENET_MEAN))
        print("Ecart-type :", [round(s, 3) for s in std], " ImageNet :", list(IMAGENET_STD))
