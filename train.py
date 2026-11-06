"""train.py : entraînement d'un modèle sur Food-11 (11 classes).

Exemples
--------
Test en petit, sans dataset (données factices) :
    python train.py --fake --model mobilenetv2 --mode frozen --subset 300 --epochs 1

Test en petit avec le vrai dataset :
    python train.py --model mobilenetv2 --mode frozen --subset 300 --epochs 1

Run réel (config 1 : MobileNetV2 gelé) :
    python train.py --model mobilenetv2 --mode frozen --epochs 10 --lr 1e-3 --out-dir results/cfg1_mnv2_frozen

Ce que le script attend des autres fichiers
-------------------------------------------
models.py (Julien) :
    get_model(name, mode, num_classes=11) -> nn.Module
data.py (Maksim) :
    get_loaders(data_dir, batch_size, img_size, subset, augment, num_workers, seed)
        -> (train_loader, val_loader, classes)
Si les signatures diffèrent, il suffit d'adapter les fonctions
build_loaders() et build_model() ci-dessous.

Fichiers produits dans --out-dir
--------------------------------
config.json           hyperparamètres utilisés
history.csv / .json   loss et accuracy train / validation, F1 macro validation, par époque
curves.png            courbes d'apprentissage
summary.json          meilleur F1, meilleure époque, nombre de paramètres, durée
val_predictions.csv   prédictions du meilleur modèle sur la validation
checkpoints/          last.pth (reprise) et best.pth (NE PAS pousser sur Git)
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

NUM_CLASSES = 11


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Entraînement Food-11")
    p.add_argument("--model", default="mobilenetv2", choices=["baseline", "mobilenetv2", "resnet18"])
    p.add_argument("--mode", default="frozen", choices=["frozen", "finetune"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--subset", type=int, default=0,
                   help="N images d'entraînement réparties sur les classes (0 = tout le jeu)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-augment", action="store_true", help="désactive l'augmentation de données")
    p.add_argument("--data-dir", default="data/food-11")
    p.add_argument("--out-dir", default="results/run")
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.9, help="momentum pour SGD")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="reprend depuis checkpoints/last.pth")
    p.add_argument("--fake", action="store_true",
                   help="données factices (torchvision FakeData) pour tester sans dataset")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Données et modèle (les deux seuls endroits qui dépendent des autres fichiers)
# ----------------------------------------------------------------------------
def build_loaders(args):
    if args.fake:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        tf = transforms.ToTensor()
        size = (3, args.img_size, args.img_size)
        n_train = args.subset or 64
        train_ds = datasets.FakeData(size=n_train, image_size=size, num_classes=NUM_CLASSES, transform=tf)
        val_ds = datasets.FakeData(size=max(n_train // 4, 16), image_size=size, num_classes=NUM_CLASSES,
                                   transform=tf, random_offset=10_000)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        return train_loader, val_loader, [str(i) for i in range(NUM_CLASSES)]

    from data import get_loaders  # écrit par Maksim (tâche 1.3)
    return get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        subset=args.subset,
        augment=not args.no_augment,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def build_model(args):
    try:
        from models import get_model  # écrit par Julien (tâche 1.5)
        return get_model(args.model, args.mode, num_classes=NUM_CLASSES)
    except ImportError:
        if not args.fake:
            raise
        # Petit CNN de secours, uniquement pour tester train.py seul avec --fake.
        print("[avertissement] models.py introuvable : CNN de secours utilisé (test --fake seulement)")
        return nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(8 * 16, NUM_CLASSES),
        )


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def set_train_mode(model):
    """model.train(), mais les BatchNorm entièrement gelées restent en mode eval
    (sinon leurs statistiques continueraient de bouger alors que le réseau est gelé)."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters()):
                m.eval()


# ----------------------------------------------------------------------------
# Boucles d'entraînement et d'évaluation
# ----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    set_train_mode(model)
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    loss_sum, n = 0.0, 0
    y_true, y_pred, conf = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        prob = torch.softmax(out.float(), dim=1)
        c, pred = prob.max(1)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        y_true += y.cpu().tolist()
        y_pred += pred.cpu().tolist()
        conf += c.cpu().tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return loss_sum / n, acc, f1, y_true, y_pred, np.array(conf)


# ----------------------------------------------------------------------------
# Sorties : historique, courbes, prédictions
# ----------------------------------------------------------------------------
HISTORY_COLS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "seconds"]


def save_history(history, out_dir):
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        w.writeheader()
        w.writerows(history)


def save_curves(history, out_dir, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set_xlabel("époque")
    ax[0].set_ylabel("loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="validation")
    ax[1].set_xlabel("époque")
    ax[1].set_ylabel("accuracy")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=150)
    plt.close(fig)


def save_val_predictions(path, classes, y_true, y_pred, conf):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_class", "pred_class", "confidence"])
        for i, (t, p, c) in enumerate(zip(y_true, y_pred, conf)):
            w.writerow([i, classes[t], classes[p], f"{c:.4f}"])


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # précision mixte seulement sur GPU
    print(f"Device : {device}" + (f" ({torch.cuda.get_device_name(0)})" if use_amp else ""))

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = build_loaders(args)
    print(f"Train : {len(train_loader.dataset)} images | Validation : {len(val_loader.dataset)} images")

    model = build_model(args).to(device)
    total, trainable = count_params(model)
    print(f"Paramètres : {total:,} au total, {trainable:,} entraînables")

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()  # softmax inclus dans la loss
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history, best_f1, best_epoch, start_epoch = [], -1.0, 0, 0

    last_path = ckpt_dir / "last.pth"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        history, best_f1, best_epoch = ck["history"], ck["best_f1"], ck["best_epoch"]
        start_epoch = ck["epoch"]
        print(f"Reprise à l'époque {start_epoch + 1} (meilleur F1 val = {best_f1:.4f})")
    else:
        if args.resume:
            print("--resume demandé mais aucun checkpoint trouvé : départ de zéro.")
        with open(out_dir / "config.json", "w") as f:
            json.dump({**vars(args), "params_total": total, "params_trainable": trainable}, f, indent=2)

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        va_loss, va_acc, va_f1, y_true, y_pred, conf = validate(model, val_loader, criterion, device, use_amp)
        secs = time.time() - t0

        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1, "seconds": secs})
        print(f"Époque {epoch + 1:>3}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} F1 {va_f1:.4f} | {secs:.0f} s")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch + 1
            best_preds = (y_true, y_pred, conf)
            torch.save({"model": model.state_dict(), "classes": classes, "model_name": args.model,
                        "mode": args.mode, "val_f1": va_f1, "epoch": epoch + 1}, ckpt_dir / "best.pth")
            save_val_predictions(out_dir / "val_predictions.csv", classes, y_true, y_pred, conf)

        # checkpoint de reprise à chaque époque
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history,
                    "best_f1": best_f1, "best_epoch": best_epoch}, last_path)
        save_history(history, out_dir)
        save_curves(history, out_dir, f"{args.model} / {args.mode}")

    summary = {
        "model": args.model, "mode": args.mode, "epochs": args.epochs,
        "best_val_f1": best_f1, "best_epoch": best_epoch,
        "params_total": total, "params_trainable": trainable,
        "train_seconds_this_session": time.time() - t_start,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Terminé. Meilleur F1 macro validation : {best_f1:.4f} (époque {best_epoch}). Sorties : {out_dir}/")


if __name__ == "__main__":
    main()
