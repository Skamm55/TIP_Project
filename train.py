"""
train.py : entraînement d'un modèle sur Food-11.
Responsable : Greg (tâche 1.6)

Le principe est le même que dans le TD : on entraîne par mini-batchs avec
l'entropie croisée, et à chaque époque on regarde la loss, l'accuracy et le
F1 macro sur la validation. On garde le modèle qui a le meilleur F1 macro.

Exemples :
    test rapide sur CPU (300 images, 1 époque) :
        python train.py --model mobilenetv2 --mode frozen --subset 300 --epochs 1
    vrai entraînement :
        python train.py --model mobilenetv2 --mode frozen --epochs 10

Résultats dans results/<modèle>_<mode>/ (ou --out-dir) :
    config.json          paramètres du run, meilleure époque, durée
    history.csv          loss, accuracy et F1 à chaque époque
    curves.png           courbes d'apprentissage
    metrics.json         métriques du meilleur modèle (fait par evaluate.py)
    val_predictions.csv  prédictions du meilleur modèle sur la validation
    best.pth             poids du meilleur modèle (ignoré par Git)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre juste les figures
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from data import DEFAULT_DATA_DIR, get_dataloaders
from evaluate import evaluer
from models import count_params, get_model

ROOT = Path(__file__).parent


def choisir_lr(model_name: str, mode: str) -> float:
    """Learning rate par défaut : plus petit en fine-tuning pour ne pas abîmer les poids ImageNet."""
    # la baseline part de zéro, il n'y a pas de poids pré-entraînés à préserver
    if mode == "finetune" and model_name != "baseline":
        return 1e-4
    return 1e-3


def dossier_sortie(args) -> Path:
    """Dossier où on enregistre les résultats du run, par exemple results/mobilenetv2_frozen."""
    if args.out_dir is not None:
        return ROOT / args.out_dir
    nom = f"{args.model}_{args.mode}"
    if args.model == "baseline":
        nom = "baseline"  # le mode ne change rien pour la baseline
    if args.no_augment:
        nom += "_noaug"
    return ROOT / "results" / nom


def mode_entrainement(model, args) -> None:
    """Passe le modèle en mode train.

    En mode frozen, le réseau pré-entraîné sert seulement d'extracteur de features :
    on remet donc ses BatchNorm en mode eval, sinon leurs moyennes et variances
    continueraient de changer avec nos images alors que le réseau est censé être gelé.
    """
    model.train()
    if args.mode == "frozen" and args.model != "baseline":
        for couche in model.modules():
            if isinstance(couche, nn.BatchNorm2d):
                couche.eval()


def entrainer_une_epoque(model, loader, criterion, optimizer, device, args):
    """Une passe sur tout le train. Renvoie la loss moyenne et l'accuracy."""
    mode_entrainement(model, args)
    total_loss, bonnes, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        sorties = model(images)
        loss = criterion(sorties, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        bonnes += (sorties.argmax(dim=1) == labels).sum().item()
        n += len(labels)
    return total_loss / n, bonnes / n


def valider(model, loader, criterion, device):
    """Passe sur la validation sans entraîner. Renvoie la loss, l'accuracy et le F1 macro."""
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            sorties = model(images)
            total_loss += criterion(sorties, labels).item() * len(labels)
            n += len(labels)
            y_true += labels.cpu().tolist()
            y_pred += sorties.argmax(dim=1).cpu().tolist()

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return total_loss / n, accuracy, f1


def sauver_historique(historique: list[dict], dossier: Path) -> None:
    """Écrit history.csv (une ligne par époque)."""
    with open(dossier / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(historique[0].keys()))
        writer.writeheader()
        writer.writerows(historique)


def tracer_courbes(historique: list[dict], dossier: Path, titre: str) -> None:
    """Courbes d'apprentissage : loss à gauche, accuracy et F1 macro à droite."""
    epoques = [h["epoque"] for h in historique]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epoques, [h["train_loss"] for h in historique], "o-", label="train")
    ax1.plot(epoques, [h["val_loss"] for h in historique], "o-", label="validation")
    ax1.set_xlabel("époque")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epoques, [h["train_acc"] for h in historique], "o-", label="accuracy train")
    ax2.plot(epoques, [h["val_acc"] for h in historique], "o-", label="accuracy validation")
    ax2.plot(epoques, [h["val_f1"] for h in historique], "o--", label="F1 macro validation")
    ax2.set_xlabel("époque")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(dossier / "curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Entraînement sur Food-11")
    parser.add_argument("--model", default="mobilenetv2", choices=["baseline", "mobilenetv2", "resnet18"])
    parser.add_argument("--mode", default="frozen", choices=["frozen", "finetune"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None,
                        help="par défaut 1e-3 (frozen et baseline) ou 1e-4 (finetune)")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"],
                        help="sgd = descente de gradient avec momentum 0.9")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset", type=int, default=None,
                        help="nombre d'images à prendre (train + validation), pour tester vite")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--no-augment", action="store_true", help="pas d'augmentation de données")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=None, help="par défaut results/<modèle>_<mode>")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.lr is None:
        args.lr = choisir_lr(args.model, args.mode)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device :", device)

    dossier = dossier_sortie(args)
    dossier.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers,
        subset=args.subset, augment=not args.no_augment, seed=args.seed,
    )
    print(f"Train : {len(train_loader.dataset)} images, validation : {len(val_loader.dataset)} images")

    model = get_model(args.model, args.mode, num_classes=len(classes), img_size=args.img_size)
    model.to(device)
    print(f"Paramètres : {count_params(model):,} au total, {count_params(model, True):,} entraînés")

    # on ne donne à l'optimiseur que les poids qui ne sont pas gelés
    a_entrainer = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(a_entrainer, lr=args.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(a_entrainer, lr=args.lr)
    criterion = nn.CrossEntropyLoss()  # contient déjà la softmax

    historique = []
    meilleur_f1, meilleure_epoque = -1.0, 0
    debut = time.time()

    for epoque in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = entrainer_une_epoque(model, train_loader, criterion, optimizer, device, args)
        val_loss, val_acc, val_f1 = valider(model, val_loader, criterion, device)

        historique.append({
            "epoque": epoque,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "secondes": round(time.time() - t0),
        })
        print(f"Époque {epoque}/{args.epochs} : train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} F1 {val_f1:.3f} | {time.time() - t0:.0f} s")

        # on garde le modèle qui a le meilleur F1 macro sur la validation
        if val_f1 > meilleur_f1:
            meilleur_f1, meilleure_epoque = val_f1, epoque
            torch.save({"model": model.state_dict(), "model_name": args.model, "mode": args.mode,
                        "epoque": epoque}, dossier / "best.pth")

        # on sauvegarde à chaque époque pour voir les courbes pendant l'entraînement
        sauver_historique(historique, dossier)
        tracer_courbes(historique, dossier, f"{args.model} {args.mode}")

    duree = (time.time() - debut) / 60
    print(f"\nMeilleur F1 macro : {meilleur_f1:.4f} à l'époque {meilleure_epoque} ({duree:.1f} min)")

    with open(dossier / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "meilleure_epoque": meilleure_epoque,
                   "meilleur_f1_val": round(meilleur_f1, 4), "duree_min": round(duree, 1)}, f, indent=2)

    # évaluation complète du meilleur modèle avec evaluate.py (métriques par classe,
    # matrice de confusion, frugalité, prédictions image par image)
    print("\nÉvaluation du meilleur modèle :")
    model.load_state_dict(torch.load(dossier / "best.pth", map_location=device)["model"])
    evaluer(model, val_loader, device, dossier)
    print(f"\nRésultats dans {dossier}")


if __name__ == "__main__":
    main()
