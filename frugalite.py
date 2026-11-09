"""
frugalite.py : tableau de frugalité des runs et figure F1 / nombre de paramètres.
Tâche 4.1

Lit results/<run>/metrics.json pour chaque run et produit :
- results/frugalite.csv : F1, paramètres, pénalité, frugalité et rangs de chaque run
- results/figures/f1_vs_parametres.png : F1 de validation en fonction du nombre de paramètres

Score du challenge : Frug = F1 - w * R / (R + 1), avec w = 0,3 et R en millions de paramètres
(c'est la même chose que F1 - w / (1 + 1/R), la forme écrite dans evaluate.py).

Usage : python frugalite.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre, on enregistre juste la figure
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

W = 0.3  # poids de la pénalité dans le score du challenge
RETENU = "mobilenetv2_finetune"  # modèle proposé pour la soumission

# nom du dossier -> nom affiché sur la figure
RUNS = {
    "baseline": "Baseline (TD)",
    "mobilenetv2_frozen": "MobileNetV2 gelé",
    "resnet18_frozen": "ResNet-18 gelé",
    "mobilenetv2_finetune": "MobileNetV2 fine-tuning",
    "resnet18_finetune": "ResNet-18 fine-tuning",
    "mobilenetv2_finetune_noaug": "MobileNetV2 fine-tuning sans augmentation",
}


def penalite(R, w=W):
    """Pénalité du score de frugalité pour R paramètres (dans l'unité choisie)."""
    return w * R / (R + 1)


def charger_metriques() -> pd.DataFrame:
    """Lit F1, nombre de paramètres et frugalité dans le metrics.json de chaque run."""
    lignes = []
    for run in RUNS:
        fichier = RESULTS_DIR / run / "metrics.json"
        if not fichier.exists():
            print(f"Run {run} ignoré : {fichier.relative_to(ROOT)} n'existe pas")
            continue
        with open(fichier, encoding="utf-8") as f:
            m = json.load(f)
        lignes.append({"run": run, "f1_val": m["f1_macro"], "n_params": m["n_params"],
                       "frugalite_json": m["frugalite"]})
    return pd.DataFrame(lignes)


def tableau(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcule la frugalité (R en millions puis R brut) et ajoute les rangs."""
    df = df.copy()
    df["R_millions"] = df["n_params"] / 1e6
    df["penalite"] = penalite(df["R_millions"])
    df["frugalite"] = df["f1_val"] - df["penalite"]

    # vérification avec la valeur calculée par evaluate.py
    for _, ligne in df.iterrows():
        if abs(ligne["frugalite"] - ligne["frugalite_json"]) > 1e-4:
            print(f"Attention : frugalité de {ligne['run']} recalculée {ligne['frugalite']:.4f}, "
                  f"mais {ligne['frugalite_json']:.4f} dans metrics.json")

    # si R était le nombre brut de paramètres, la pénalité vaudrait presque 0,3 pour tous
    df["frugalite_R_brut"] = df["f1_val"] - penalite(df["n_params"])

    # rang 1 = meilleur
    df["rang_frugalite"] = df["frugalite"].rank(ascending=False).astype(int)
    df["rang_f1"] = df["f1_val"].rank(ascending=False).astype(int)
    df["rang_R_brut"] = df["frugalite_R_brut"].rank(ascending=False).astype(int)

    colonnes = ["run", "f1_val", "n_params", "R_millions", "penalite", "frugalite", "rang_frugalite",
                "rang_f1", "frugalite_R_brut", "rang_R_brut"]
    return df[colonnes]  # pas d'arrondi ici, sinon 0,70348 -> 0,7035 -> 0,704 sur la figure


def figure(df: pd.DataFrame) -> None:
    """F1 de validation en fonction du nombre de paramètres, avec des courbes d'iso-frugalité."""
    fig, ax = plt.subplots(figsize=(9, 6))
    y_min, y_max = 0.80, 0.95

    # courbes d'iso-frugalité : tous les modèles sur une courbe ont le même score
    R = [0.8 * 1.02 ** k for k in range(165)]  # de 0,8 à environ 21 M, régulier en échelle log
    for frug in (0.60, 0.65, 0.70):
        f1 = [frug + penalite(r) for r in R]
        ax.plot(R, f1, ":", color="gray", linewidth=1)
        # étiquette au point le plus à droite de la courbe encore dans le cadre
        dedans = [k for k, y in enumerate(f1) if y <= y_max - 0.015]
        i = dedans[-1] if dedans else 0
        ax.annotate(f"Frug = {frug:.2f}".replace(".", ","), xy=(R[i], f1[i]), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color="gray")

    for _, ligne in df.iterrows():
        nom = RUNS[ligne["run"]]
        x, y = ligne["R_millions"], ligne["f1_val"]
        if y < y_min:
            # la baseline sort du cadre : on l'indique par une flèche en bas de la figure
            ax.annotate(f"baseline : F1 {y:.2f}".replace(".", ","), xy=(x, y_min), xytext=(x, y_min + 0.02),
                        ha="center", fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
            continue
        if ligne["run"] == RETENU:
            ax.plot(x, y, "*", color="tab:red", markersize=16, zorder=3)
            texte = f"{nom} (retenu)\nFrug = {ligne['frugalite']:.3f}".replace(".", ",")
            ax.annotate(texte, xy=(x, y), xytext=(10, -4), textcoords="offset points", fontsize=9,
                        fontweight="bold", color="tab:red")
        else:
            ax.plot(x, y, "o", color="tab:blue", zorder=3)
            texte = f"{nom}\nFrug = {ligne['frugalite']:.3f}".replace(".", ",")
            ax.annotate(texte, xy=(x, y), xytext=(8, -4), textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(0.8, 40)  # de la place à droite pour les noms
    # graduations en nombres simples plutôt qu'en puissances de 10
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("nombre de paramètres (millions, échelle log)")
    ax.set_ylabel("F1 macro de validation")
    ax.set_title("F1 de validation et nombre de paramètres (pointillés : même score de frugalité)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "f1_vs_parametres.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metriques = charger_metriques()
    if metriques.empty:
        raise SystemExit("Aucun metrics.json trouvé dans results/")

    resultats = tableau(metriques)
    figure(resultats)
    resultats = resultats.round(4)
    resultats.to_csv(RESULTS_DIR / "frugalite.csv", index=False)

    print(resultats.to_string(index=False))
    print("\nTableau dans results/frugalite.csv, figure dans results/figures/f1_vs_parametres.png")
