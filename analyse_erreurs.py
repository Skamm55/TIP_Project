"""
analyse_erreurs.py : matrice de confusion et analyse des erreurs du meilleur modèle.
Responsable : Oggy (tâche 4.3)

Tout est calculé depuis results/<run>/val_predictions.csv, sans les poids.
Produit dans results/figures/ :
- confusion_<run>.png : matrice de confusion normalisée par ligne (rappel sur la diagonale)
- f1_par_classe_<run>.png : précision, rappel et F1 par classe
- erreurs_<run>.png : planche d'images mal classées (seulement avec --images, il faut le dataset)

Usage :
    python analyse_erreurs.py
    python analyse_erreurs.py --images "meat/579.jpg,dairy/538.jpg"   (planche, besoin de data/food-11)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre juste les figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data import CLASSES, DEFAULT_DATA_DIR

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def plot_confusion(df, run):
    """Matrice de confusion normalisée par ligne, avec le nombre d'images dans chaque case."""
    matrice = confusion_matrix(df["vraie_classe"], df["classe_predite"], labels=CLASSES)
    # normalisée par ligne : la case (i, j) = part des images de la classe i prédites j
    norm = matrice / matrice.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if matrice[i, j] > 0:
                couleur = "white" if norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{norm[i, j]:.2f}\n({matrice[i, j]})", ha="center", va="center",
                        fontsize=7, color=couleur)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title(f"Matrice de confusion ({run}, validation)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_{run}.png", dpi=150)
    plt.close()
    return matrice


def plot_f1_par_classe(df, run):
    """Précision, rappel et F1 par classe, triés par F1. Renvoie aussi le tableau des scores."""
    precision, rappel, f1, support = precision_recall_fscore_support(
        df["vraie_classe"], df["classe_predite"], labels=CLASSES, zero_division=0)
    ordre = np.argsort(f1)
    y = np.arange(len(CLASSES))

    plt.figure(figsize=(8, 6))
    plt.barh(y - 0.27, precision[ordre], height=0.27, label="précision", color="tab:blue")
    plt.barh(y, rappel[ordre], height=0.27, label="rappel", color="tab:orange")
    plt.barh(y + 0.27, f1[ordre], height=0.27, label="F1", color="tab:green")
    noms = [f"{CLASSES[i]} ({support[i]})" for i in ordre]
    plt.yticks(y, noms)
    plt.xlim(max(0, min(precision.min(), rappel.min()) - 0.05), 1)
    plt.xlabel("Score")
    plt.title(f"Scores par classe ({run}, validation)\nentre parenthèses : nombre d'images")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"f1_par_classe_{run}.png", dpi=150)
    plt.close()
    return pd.DataFrame({"classe": CLASSES, "precision": precision, "rappel": rappel,
                         "f1": f1, "nb_images": support}).sort_values("f1")


def plot_erreurs(df, images, data_dir, run):
    """Planche des images choisies, avec vraie classe, classe prédite et confiance."""
    lignes = df.set_index("image").loc[images]
    n = len(images)
    colonnes = 3
    rangees = (n + colonnes - 1) // colonnes
    fig, axes = plt.subplots(rangees, colonnes, figsize=(colonnes * 3.5, rangees * 3.8))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (nom, ligne) in zip(axes, lignes.iterrows()):
        with Image.open(data_dir / "train" / nom) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{nom}\nvraie : {ligne['vraie_classe']}\n"
                     f"prédite : {ligne['classe_predite']} ({ligne['confiance']:.0%})", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"erreurs_{run}.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="mobilenetv2_finetune")
    parser.add_argument("--images", help="liste d'images séparées par des virgules, ex meat/579.jpg")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    df = pd.read_csv(RESULTS_DIR / args.run / "val_predictions.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    erreurs = df[df["vraie_classe"] != df["classe_predite"]]
    print(f"Run {args.run} : {len(erreurs)} erreurs sur {len(df)} images "
          f"({len(erreurs) / len(df):.1%})\n")

    plot_confusion(df, args.run)
    scores = plot_f1_par_classe(df, args.run)
    print(scores.round(3).to_string(index=False))

    print("\nConfusions les plus fréquentes (vraie -> prédite) :")
    paires = erreurs.groupby(["vraie_classe", "classe_predite"]).size().sort_values(ascending=False)
    for (vraie, pred), nb in paires.head(10).items():
        print(f"  {vraie:16s} -> {pred:16s} {nb}")

    justes = df[df["vraie_classe"] == df["classe_predite"]]
    print(f"\nConfiance moyenne : {justes['confiance'].mean():.2f} quand c'est juste, "
          f"{erreurs['confiance'].mean():.2f} quand c'est faux")
    print(f"Erreurs avec une confiance > 90 % : {(erreurs['confiance'] > 0.9).sum()}")

    # candidats pour la planche : les erreurs les plus sûres d'elles et les plus hésitantes
    print("\nErreurs les plus sûres d'elles :")
    print(erreurs.sort_values("confiance", ascending=False).head(8).to_string(index=False))
    print("\nErreurs les plus hésitantes :")
    print(erreurs.sort_values("confiance").head(5).to_string(index=False))

    if args.images:
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        plot_erreurs(df, [i.strip() for i in args.images.split(",")], data_dir, args.run)
        print(f"\nPlanche enregistrée : results/figures/erreurs_{args.run}.png")

    print(f"\nFigures enregistrées : results/figures/confusion_{args.run}.png, "
          f"results/figures/f1_par_classe_{args.run}.png")
