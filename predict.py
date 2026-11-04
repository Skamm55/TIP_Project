"""
predict.py : génération du JSON de prédictions pour le serveur d'évaluation.
Responsable : Oggy (tâche 1.8)

Format strict attendu : {"0": "bread", "1": "dessert", ...}
- clé : nom de l'image du test sans extension
- valeur : nom de classe en minuscules, identique au nom du dossier du train

Usage :
    python predict.py --model mobilenetv2 --mode frozen --weights checkpoints/best.pth
    python predict.py --self-test
"""

import argparse
import json
from pathlib import Path

import torch

from data import CLASSES, DEFAULT_DATA_DIR, get_test_loader

ROOT = Path(__file__).parent
NB_IMAGES_TEST = 3097


def choisir_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def charger_poids(model, chemin, device="cpu"):
    """Accepte un state_dict seul ou un checkpoint avec le state_dict sous la clé "model"."""
    checkpoint = torch.load(chemin, map_location=device)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    return model


def predire_test(model, test_loader, device="cpu"):
    """Renvoie {nom de l'image: classe prédite}."""
    model.to(device)
    model.eval()
    predictions = {}
    with torch.no_grad():
        for images, noms in test_loader:
            classes = model(images.to(device)).argmax(dim=1).cpu().tolist()
            for nom, c in zip(noms, classes):
                predictions[nom] = CLASSES[c]
    return predictions


def verifier_predictions(predictions, nb_attendu=NB_IMAGES_TEST):
    if len(predictions) != nb_attendu:
        raise ValueError(f"{len(predictions)} prédictions au lieu de {nb_attendu}")
    for nom, classe in predictions.items():
        if not nom.isdigit():
            raise ValueError(f"Clé '{nom}' : il faut le nom sans extension, ex '0'")
        if classe not in CLASSES:
            raise ValueError(f"Classe '{classe}' inconnue (image {nom})")
    if sorted(int(nom) for nom in predictions) != list(range(nb_attendu)):
        raise ValueError("Il manque des images ou il y a des doublons")


def ecrire_json(predictions, chemin):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    # tri numérique : 0, 1, 2... et pas 0, 1, 10...
    tri = dict(sorted(predictions.items(), key=lambda x: int(x[0])))
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(tri, f, indent=4)


def self_test():
    """Test sans dataset : fausses images et petit modèle non entraîné."""
    import tempfile

    from PIL import Image

    n = 25
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "test").mkdir()
        for i in range(n):
            Image.new("RGB", (100 + i * 10, 120), (i * 10, 50, 200)).save(tmp / "test" / f"{i}.jpg")

        loader = get_test_loader(str(tmp), img_size=64, batch_size=8)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 64 * 64, len(CLASSES)))

        torch.save(model.state_dict(), tmp / "poids.pth")
        torch.save({"epoch": 3, "model": model.state_dict()}, tmp / "checkpoint.pth")
        charger_poids(model, tmp / "poids.pth")
        charger_poids(model, tmp / "checkpoint.pth")
        print("Test 1 OK : chargement des poids")

        predictions = predire_test(model, loader)
        verifier_predictions(predictions, nb_attendu=n)
        print(f"Test 2 OK : {len(predictions)} prédictions au bon format")

        ecrire_json(predictions, tmp / "predictions.json")
        with open(tmp / "predictions.json", encoding="utf-8") as f:
            relu = json.load(f)
        assert list(relu)[:12] == [str(i) for i in range(12)]
        print("Test 3 OK : JSON trié, début :", dict(list(relu.items())[:4]))

        for faux in [{"0.jpg": "bread"}, {"0": "Bread"}, {"0": "bread", "2": "rice"}]:
            try:
                verifier_predictions(faux, nb_attendu=len(faux))
                raise AssertionError(f"format faux accepté : {faux}")
            except ValueError:
                pass
        print("Test 4 OK : mauvais formats refusés")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="baseline, mobilenetv2 ou resnet18")
    parser.add_argument("--mode", default="finetune", help="frozen ou finetune")
    parser.add_argument("--weights", help="fichier .pth du modèle entraîné")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default="results/predictions.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        if args.model is None or args.weights is None:
            parser.error("il faut --model et --weights (ou --self-test)")

        from models import get_model  # ici pour que --self-test marche sans models.py

        device = choisir_device()
        print("Device :", device)
        model = get_model(args.model, args.mode, num_classes=len(CLASSES))
        charger_poids(model, args.weights, device)

        test_loader = get_test_loader(args.data_dir, args.img_size, args.batch_size, args.num_workers)
        predictions = predire_test(model, test_loader, device)
        verifier_predictions(predictions)
        ecrire_json(predictions, ROOT / args.out)

        print(f"{len(predictions)} prédictions écrites dans {args.out}")
        for classe in CLASSES:
            print(f"  {classe:16s} {sum(c == classe for c in predictions.values())}")
