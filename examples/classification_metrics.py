"""Mesurer classification binaire, multiclasse, multilabel et hiérarchique.

Données synthétiques explicites : démonstration de métriques, pas entraînement de CNN.
Exécution : python examples/classification_metrics.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    hamming_loss, multilabel_confusion_matrix
)

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


def binaire():
    # Matrice attendue : [[TN=55, FP=10], [FN=5, TP=30]].
    y_true = np.array([0] * 65 + [1] * 35)
    y_pred = np.array([0] * 55 + [1] * 10 + [0] * 5 + [1] * 30)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm.tolist() == [[55, 10], [5, 30]]
    print("\n== Classification binaire ==")
    print("Matrice (lignes : réalité, colonnes : prédiction) :\n", cm)
    print(f"Accuracy  = {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision = {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    = {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score  = {f1_score(y_true, y_pred):.4f}")
    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Non", "Oui"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Classification binaire — Matrice de confusion")
    fig.tight_layout()
    fig.savefig(OUT / "classification_binaire_confusion.png", dpi=160)
    plt.close(fig)


def multiclasse():
    # Trois classes exclusives : Chat=0, Chien=1, Oiseau=2.
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 1, 2, 1, 2, 2, 0, 2]
    print("\n== Classification multiclasse ==")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Macro-F1:", round(f1_score(y_true, y_pred, average="macro"), 4))
    print(classification_report(y_true, y_pred,
                                labels=[0, 1, 2],
                                target_names=["Chat", "Chien", "Oiseau"],
                                zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Chat", "Chien", "Oiseau"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Classification multiclasse")
    fig.tight_layout()
    fig.savefig(OUT / "classification_multiclasse_confusion.png", dpi=160)
    plt.close(fig)


def multilabel():
    # Chaque image peut contenir plusieurs étiquettes simultanées.
    # Colonnes : chat, chien, arbre. Les lignes représentent six images.
    y_true = np.array([
        [1, 0, 0], [0, 1, 1], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 0, 0]
    ])
    y_pred = np.array([
        [1, 0, 0], [0, 1, 0], [1, 0, 0],
        [0, 0, 1], [0, 0, 1], [0, 1, 0]
    ])
    print("\n== Classification multilabel ==")
    print("Exact match (tous les labels corrects) :", round(accuracy_score(y_true, y_pred), 4))
    print("Micro-F1 :", round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4))
    print("Macro-F1 :", round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4))
    print("Hamming loss (plus petit = mieux) :", round(hamming_loss(y_true, y_pred), 4))
    print("Matrices de confusion par label (chat, chien, arbre) :\n",
          multilabel_confusion_matrix(y_true, y_pred))


def hierarchique():
    # Démo simplifiée : prédiction d'une feuille puis d'une catégorie parente.
    parent = {"Labrador": "Chien", "Caniche": "Chien", "Persan": "Chat", "Siamois": "Chat"}
    y_true = ["Labrador", "Caniche", "Persan", "Siamois"]
    y_pred = ["Labrador", "Labrador", "Persan", "Persan"]
    feuille = accuracy_score(y_true, y_pred)
    niveau_parent = accuracy_score([parent[c] for c in y_true], [parent[c] for c in y_pred])
    print("\n== Classification hiérarchique ==")
    print(f"Accuracy classe fine : {feuille:.2f} ; accuracy famille : {niveau_parent:.2f}")
    print("Cet exemple illustre une évaluation par niveau, pas une métrique hiérarchique universelle.")


if __name__ == "__main__":
    binaire()
    multiclasse()
    multilabel()
    hierarchique()
    print("\nFigure(s) enregistrée(s) dans :", OUT)
