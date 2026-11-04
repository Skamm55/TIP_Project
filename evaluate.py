"""
evaluate.py : métriques d'évaluation.
Responsable : Oggy (tâche 1.7)

- F1 macro, précision et rappel par classe, matrice de confusion
- score de frugalité : Frug = P - w / (1 + 1/R), w = 0.3, R en millions de paramètres
- export metrics.json et val_predictions.csv

Test sans dataset : python evaluate.py
"""

import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from data import CLASSES

RESULTS_DIR = Path(__file__).parent / "results"


def predire(model, loader, device="cpu"):
    """Renvoie les vraies classes, les classes prédites et la confiance (softmax)."""
    model.to(device)
    model.eval()
    y_true, y_pred, confiances = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probas = torch.softmax(model(images.to(device)), dim=1)
            conf, pred = probas.max(dim=1)
            y_true += labels.tolist()
            y_pred += pred.cpu().tolist()
            confiances += conf.cpu().tolist()
    return y_true, y_pred, confiances


def noms_images(loader):
    """Noms des images du loader de validation, dans le même ordre que les prédictions
    (marche seulement avec shuffle=False)."""
    subset = loader.dataset
    noms = []
    for i in subset.indices:
        chemin = Path(subset.dataset.samples[i][0])
        noms.append(f"{chemin.parent.name}/{chemin.name}")
    return noms


def frugalite(P, R, w=0.3):
    """R en millions de paramètres (c'est ce qui redonne les exemples de l'énoncé)."""
    return P - w / (1 + 1 / R)


def compter_params(model):
    return sum(p.numel() for p in model.parameters())


def calculer_metriques(y_true, y_pred, n_params=None, w=0.3):
    labels = list(range(len(CLASSES)))
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rappels = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    matrice = confusion_matrix(y_true, y_pred, labels=labels)

    metriques = {
        "f1_macro": float(f1_macro),
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
        "par_classe": {},
        "matrice_confusion": matrice.tolist(),  # lignes : vraies classes
        "classes": CLASSES,
    }
    for i, classe in enumerate(CLASSES):
        metriques["par_classe"][classe] = {
            "precision": float(precisions[i]),
            "rappel": float(rappels[i]),
            "f1": float(f1s[i]),
            "nb_images": int(matrice[i].sum()),
        }

    if n_params is not None:
        metriques["n_params"] = n_params
        metriques["frugalite"] = frugalite(f1_macro, n_params / 1e6, w)
    return metriques


def sauver_resultats(metriques, y_true, y_pred, confiances, noms, dossier=RESULTS_DIR):
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    with open(dossier / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metriques, f, indent=2, ensure_ascii=False)

    with open(dossier / "val_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "vraie_classe", "classe_predite", "confiance"])
        for nom, t, p, c in zip(noms, y_true, y_pred, confiances):
            writer.writerow([nom, CLASSES[t], CLASSES[p], round(c, 4)])


def afficher(metriques):
    print(f"F1 macro : {metriques['f1_macro']:.4f}   accuracy : {metriques['accuracy']:.4f}")
    if "frugalite" in metriques:
        print(f"Paramètres : {metriques['n_params'] / 1e6:.2f} M   frugalité : {metriques['frugalite']:.4f}")
    print(f"\n{'classe':16s} {'précision':>9s} {'rappel':>7s} {'f1':>6s} {'n':>5s}")
    for classe, m in metriques["par_classe"].items():
        print(f"{classe:16s} {m['precision']:9.3f} {m['rappel']:7.3f} {m['f1']:6.3f} {m['nb_images']:5d}")


def evaluer(model, val_loader, device="cpu", dossier=RESULTS_DIR, w=0.3):
    """Prédictions + métriques + export + affichage."""
    y_true, y_pred, confiances = predire(model, val_loader, device)
    metriques = calculer_metriques(y_true, y_pred, compter_params(model), w)
    sauver_resultats(metriques, y_true, y_pred, confiances, noms_images(val_loader), dossier)
    afficher(metriques)
    return metriques


if __name__ == "__main__":
    import random
    import tempfile

    from torch.utils.data import DataLoader, TensorDataset

    random.seed(0)
    torch.manual_seed(0)
    n = len(CLASSES)

    # tout juste -> F1 = 1
    y_true = [i % n for i in range(110)]
    m = calculer_metriques(y_true, y_true)
    assert abs(m["f1_macro"] - 1) < 1e-9
    assert m["matrice_confusion"][0][0] == 10
    print("Test 1 OK : F1 = 1 quand tout est juste")

    # au hasard -> F1 proche de 1/11
    y_pred = [random.randrange(n) for _ in y_true]
    m = calculer_metriques(y_true, y_pred)
    assert m["f1_macro"] < 0.3
    print(f"Test 2 OK : F1 au hasard = {m['f1_macro']:.3f}")

    # exemples et question de l'énoncé
    assert round(frugalite(0.80, 3.5), 2) == 0.57
    assert round(frugalite(0.82, 11.6), 2) == 0.54
    f1_min = frugalite(0.8, 3.5) + 0.3 / (1 + 1 / 11.6)
    assert abs(f1_min - 0.843) < 0.001
    print(f"Test 3 OK : frugalité conforme à l'énoncé, F1 minimum à 11,6 M = {f1_min:.3f}")

    # petit modèle sur de fausses images
    loader = DataLoader(TensorDataset(torch.randn(20, 3, 32, 32), torch.randint(0, n, (20,))), batch_size=8)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, n))
    y_true, y_pred, conf = predire(model, loader)
    assert len(y_true) == len(y_pred) == len(conf) == 20
    assert all(0 < c <= 1 for c in conf)
    print("Test 4 OK : predire() sur 20 fausses images")

    m = calculer_metriques(y_true, y_pred, compter_params(model))
    with tempfile.TemporaryDirectory() as tmp:
        sauver_resultats(m, y_true, y_pred, conf, [f"faux/{i}.jpg" for i in range(20)], tmp)
        with open(Path(tmp) / "metrics.json", encoding="utf-8") as f:
            assert "f1_macro" in json.load(f)
        with open(Path(tmp) / "val_predictions.csv", encoding="utf-8") as f:
            assert len(f.readlines()) == 21
    print("Test 5 OK : metrics.json et val_predictions.csv\n")

    afficher(m)
