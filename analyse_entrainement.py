"""
analyse_entrainement.py : courbes et synthèse des entraînements.
Responsable : Maksim (tâches 4.2 et 4.4)

Lit results/<run>/history.csv pour chaque run et produit :
- results/figures/f1_validation_runs.png : F1 macro de validation de tous les runs
- results/figures/train_vs_val_finetuning.png : train vs validation en fine-tuning
- results/figures/train_vs_val_gele.png : train vs validation en mode gelé
- results/figures/augmentation_avec_sans.png : MobileNetV2 fine-tuning avec et sans augmentation (4.4)
- results/figures/augmentation_f1_par_classe.png : F1 par classe avec et sans augmentation (4.4)
- results/synthese_runs.csv : une ligne par run (aussi affichée dans la console)

Usage : python analyse_entrainement.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre juste les figures
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# nom du dossier -> (nom dans les légendes, couleur, style de trait)
# même couleur pour un même réseau, plein en fine-tuning, pointillé en gelé
RUNS = {
    "baseline": ("Baseline (TD)", "gray", "-"),
    "mobilenetv2_frozen": ("MobileNetV2 gelé", "tab:blue", "--"),
    "resnet18_frozen": ("ResNet-18 gelé", "tab:orange", "--"),
    "mobilenetv2_finetune": ("MobileNetV2 fine-tuning", "tab:blue", "-"),
    "resnet18_finetune": ("ResNet-18 fine-tuning", "tab:orange", "-"),
    "mobilenetv2_finetune_noaug": ("MobileNetV2 fine-tuning sans augmentation", "tab:blue", ":"),
}


def charger_historiques() -> dict[str, pd.DataFrame]:
    """Lit le history.csv de chaque run. Les runs absents sont ignorés avec un message."""
    historiques = {}
    for run in RUNS:
        fichier = RESULTS_DIR / run / "history.csv"
        if not fichier.exists():
            print(f"Run {run} ignoré : {fichier.relative_to(ROOT)} n'existe pas")
            continue
        historiques[run] = pd.read_csv(fichier)
    return historiques


def figure_f1(historiques: dict[str, pd.DataFrame]) -> None:
    """F1 macro de validation à chaque époque, une courbe par run (sans le run _noaug)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for run, h in historiques.items():
        if run.endswith("_noaug"):
            continue  # comparé à part dans figure_augmentation (tâche 4.4)
        nom, couleur, trait = RUNS[run]
        ax.plot(h["epoque"], h["val_f1"], trait, color=couleur, label=nom)
        # un point sur le meilleur F1 du run
        meilleur = h.loc[h["val_f1"].idxmax()]
        ax.plot(meilleur["epoque"], meilleur["val_f1"], "o", color=couleur)

    ax.set_xlabel("époque")
    ax.set_xticks(range(0, 21, 2))  # époques entières (pas de 2.5, 7.5...)
    ax.set_ylabel("F1 macro de validation")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("F1 macro de validation au fil des époques (le point marque le meilleur F1)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "f1_validation_runs.png", dpi=150)
    plt.close(fig)


def figure_train_vs_val(historiques: dict[str, pd.DataFrame], runs: list[str], nom_fichier: str,
                        titre: str) -> None:
    """Accuracy (en haut) et loss (en bas), train vs validation, une colonne par run."""
    runs = [r for r in runs if r in historiques]
    if not runs:
        print(f"{nom_fichier} non créée : aucun des runs {runs} n'est disponible")
        return

    fig, axes = plt.subplots(2, len(runs), figsize=(6 * len(runs), 8), squeeze=False)
    for col, run in enumerate(runs):
        h = historiques[run]
        nom, couleur, _ = RUNS[run]
        ax_acc, ax_loss = axes[0, col], axes[1, col]

        ax_acc.plot(h["epoque"], h["train_acc"], "o-", color=couleur, alpha=0.5, label="train")
        ax_acc.plot(h["epoque"], h["val_acc"], "o-", color=couleur, label="validation")
        ax_acc.set_title(nom)
        ax_acc.set_ylabel("accuracy")

        ax_loss.plot(h["epoque"], h["train_loss"], "o-", color=couleur, alpha=0.5, label="train")
        ax_loss.plot(h["epoque"], h["val_loss"], "o-", color=couleur, label="validation")
        ax_loss.set_xlabel("époque")
        ax_loss.set_ylabel("loss")

        # ligne verticale à l'époque où la loss de validation est la plus basse
        epoque_min = h.loc[h["val_loss"].idxmin(), "epoque"]
        for ax in (ax_acc, ax_loss):
            ax.axvline(epoque_min, color="black", linestyle=":", linewidth=1)
            ax.grid(alpha=0.3)
            ax.legend()
        # texte à gauche de la ligne si elle est dans la 2e moitié de l'axe, sinon il serait coupé
        a_gauche = epoque_min > (h["epoque"].min() + h["epoque"].max()) / 2
        ax_loss.annotate("min loss val", xy=(epoque_min, ax_loss.get_ylim()[1]),
                         xytext=(-4 if a_gauche else 4, -14), textcoords="offset points",
                         ha="right" if a_gauche else "left", fontsize=9)

    fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / nom_fichier, dpi=150)
    plt.close(fig)


def figure_augmentation(historiques: dict[str, pd.DataFrame]) -> None:
    """Tâche 4.4 : MobileNetV2 fine-tuning avec et sans augmentation, accuracy à gauche, loss à droite."""
    runs = {"mobilenetv2_finetune": ("avec", "tab:blue"), "mobilenetv2_finetune_noaug": ("sans", "tab:red")}
    if not all(r in historiques for r in runs):
        print("augmentation_avec_sans.png non créée : il manque un des deux runs MobileNetV2 fine-tuning")
        return

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))
    for run, (nom, couleur) in runs.items():
        h = historiques[run]
        # trait plein pour la validation, pointillé pour le train
        ax_acc.plot(h["epoque"], h["val_acc"], "-", color=couleur, label=f"validation {nom} augmentation")
        ax_acc.plot(h["epoque"], h["train_acc"], "--", color=couleur, label=f"train {nom} augmentation")
        ax_loss.plot(h["epoque"], h["val_loss"], "-", color=couleur, label=f"validation {nom} augmentation")
        ax_loss.plot(h["epoque"], h["train_loss"], "--", color=couleur, label=f"train {nom} augmentation")

    # une ligne verticale au minimum de la loss de validation de chaque run (après avoir tracé les
    # courbes, pour que get_ylim donne déjà les bonnes limites)
    for run, (nom, couleur) in runs.items():
        h = historiques[run]
        epoque_min = h.loc[h["val_loss"].idxmin(), "epoque"]
        ax_loss.axvline(epoque_min, color=couleur, linestyle=":", linewidth=1)
        # texte à gauche de la ligne si elle est dans la 2e moitié de l'axe, sinon il serait coupé
        a_gauche = epoque_min > (h["epoque"].min() + h["epoque"].max()) / 2
        ax_loss.annotate(f"min {nom}", xy=(epoque_min, ax_loss.get_ylim()[1]),
                         xytext=(-4 if a_gauche else 4, -14), textcoords="offset points",
                         ha="right" if a_gauche else "left", fontsize=9, color=couleur)

    ax_acc.set_title("Accuracy")
    ax_acc.set_ylabel("accuracy")
    ax_loss.set_title("Loss")
    ax_loss.set_ylabel("loss")
    for ax in (ax_acc, ax_loss):
        ax.set_xlabel("époque")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("MobileNetV2 fine-tuning : avec et sans augmentation de données")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "augmentation_avec_sans.png", dpi=150)
    plt.close(fig)


def figure_f1_par_classe_augmentation() -> None:
    """Tâche 4.4 : F1 par classe avec et sans augmentation, classes triées par écart décroissant."""
    f1 = {}
    for run, nom in [("mobilenetv2_finetune", "avec"), ("mobilenetv2_finetune_noaug", "sans")]:
        fichier = RESULTS_DIR / run / "metrics.json"
        if not fichier.exists():
            print(f"augmentation_f1_par_classe.png non créée : {fichier.relative_to(ROOT)} n'existe pas")
            return
        with open(fichier, encoding="utf-8") as f:
            f1[nom] = {c: v["f1"] for c, v in json.load(f)["par_classe"].items()}

    tableau = pd.DataFrame(f1)
    tableau["ecart"] = tableau["avec"] - tableau["sans"]  # > 0 : la classe perd sans augmentation
    tableau = tableau.sort_values("ecart", ascending=False)

    x = range(len(tableau))
    largeur = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - largeur / 2 for i in x], tableau["avec"], largeur, color="tab:blue", label="avec augmentation")
    ax.bar([i + largeur / 2 for i in x], tableau["sans"], largeur, color="tab:red", label="sans augmentation")
    # l'écart au-dessus de chaque paire de barres
    for i, (_, ligne) in zip(x, tableau.iterrows()):
        ecart = round(-ligne["ecart"], 3) + 0.0  # le + 0.0 évite d'afficher "-0,000" pour rice
        texte = f"{ecart:+.3f}".replace(".", ",") if ecart != 0 else "0,000"
        ax.annotate(texte, xy=(i, max(ligne["avec"], ligne["sans"])),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tableau.index, rotation=30, ha="right")
    ax.set_ylabel("F1 de validation")
    ax.set_ylim(0.8, 1.03)  # tous les F1 sont au-dessus de 0,8 : on zoome pour voir les écarts
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", ncol=2)  # en haut à gauche, la place est libre au-dessus de dairy et egg
    ax.set_title("F1 par classe avec et sans augmentation (écart sans - avec au-dessus des barres)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "augmentation_f1_par_classe.png", dpi=150)
    plt.close(fig)


def synthese(historiques: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une ligne de chiffres clés par run."""
    lignes = []
    for run, h in historiques.items():
        meilleur = h.loc[h["val_f1"].idxmax()]
        plus_basse = h.loc[h["val_loss"].idxmin()]
        derniere = h.iloc[-1]
        # la 1re époque est plus lente (chargement initial des images), on la retire de la moyenne
        secondes = h["secondes"].iloc[1:] if len(h) > 1 else h["secondes"]

        lignes.append({
            "run": run,
            "epoques": len(h),
            "meilleur_f1_val": round(meilleur["val_f1"], 4),
            "epoque_meilleur_f1": int(meilleur["epoque"]),
            "acc_train_finale": round(derniere["train_acc"], 4),
            "acc_val_finale": round(derniere["val_acc"], 4),
            "ecart_train_val": round(100 * (derniere["train_acc"] - derniere["val_acc"]), 1),
            "epoque_min_loss_val": int(plus_basse["epoque"]),
            "loss_val_min": round(plus_basse["val_loss"], 4),
            "loss_val_finale": round(derniere["val_loss"], 4),
            "duree_moyenne_epoque_s": round(secondes.mean(), 1),
        })
    return pd.DataFrame(lignes)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    historiques = charger_historiques()
    if not historiques:
        raise SystemExit("Aucun run trouvé dans results/")

    figure_f1(historiques)
    figure_train_vs_val(historiques, ["mobilenetv2_finetune", "resnet18_finetune"],
                        "train_vs_val_finetuning.png", "Fine-tuning : train vs validation")
    figure_train_vs_val(historiques, ["mobilenetv2_frozen", "resnet18_frozen"],
                        "train_vs_val_gele.png", "Mode gelé : train vs validation")
    figure_augmentation(historiques)
    figure_f1_par_classe_augmentation()

    tableau = synthese(historiques)
    tableau.to_csv(RESULTS_DIR / "synthese_runs.csv", index=False)
    print(tableau.to_string(index=False))
    print("\nFigures dans results/figures/, tableau dans results/synthese_runs.csv")
