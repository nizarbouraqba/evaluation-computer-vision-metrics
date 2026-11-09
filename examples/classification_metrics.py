"""Projet 1 : reconnaître de vrais chiffres manuscrits avec une régression logistique."""
import argparse
import hashlib
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .commun import RACINE, environnement, sauver_json


def executer(sortie=RACINE / 'resultats', graine=42):
    debut = time.perf_counter()
    sortie = Path(sortie) / 'classification'
    sortie.mkdir(parents=True, exist_ok=True)
    chiffres = load_digits()
    indices = np.arange(len(chiffres.target))
    apprentissage, test = train_test_split(indices, test_size=0.25, random_state=graine,
                                          stratify=chiffres.target)
    x_train, x_test = chiffres.data[apprentissage], chiffres.data[test]
    y_train, y_test = chiffres.target[apprentissage], chiffres.target[test]
    # Le scaler est ajusté uniquement sur l'apprentissage : pas de fuite du test.
    modele = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                                random_state=graine))
    modele.fit(x_train, y_train)
    predictions = modele.predict(x_test)
    probabilites = modele.predict_proba(x_test)
    multiclasse = {'accuracy': float(accuracy_score(y_test, predictions)),
                  'precision_macro': float(precision_score(y_test, predictions, average='macro', zero_division=0)),
                  'rappel_macro': float(recall_score(y_test, predictions, average='macro', zero_division=0)),
                  'f1_macro': float(f1_score(y_test, predictions, average='macro', zero_division=0)),
                  'rapport_par_classe': classification_report(y_test, predictions, output_dict=True, zero_division=0),
                  'matrice_confusion': confusion_matrix(y_test, predictions).tolist()}
    # Deuxième question réelle : le chiffre photographié est-il pair ?
    pair_train, pair_test = (y_train % 2 == 0).astype(int), (y_test % 2 == 0).astype(int)
    modele_pair = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=graine))
    modele_pair.fit(x_train, pair_train)
    proba_pair = modele_pair.predict_proba(x_test)[:, 1]
    pred_pair = (proba_pair >= 0.5).astype(int)
    binaire = {'classe_positive': 'chiffre pair', 'seuil': 0.5,
               'accuracy': float(accuracy_score(pair_test, pred_pair)),
               'precision': float(precision_score(pair_test, pred_pair, zero_division=0)),
               'rappel': float(recall_score(pair_test, pred_pair, zero_division=0)),
               'f1': float(f1_score(pair_test, pred_pair, zero_division=0)),
               'roc_auc': float(roc_auc_score(pair_test, proba_pair)),
               'average_precision': float(average_precision_score(pair_test, proba_pair)),
               'matrice_confusion': confusion_matrix(pair_test, pred_pair).tolist()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=axes[0], colorbar=False)
    ConfusionMatrixDisplay.from_predictions(pair_test, pred_pair, ax=axes[1],
                                           display_labels=['impair', 'pair'], colorbar=False)
    axes[0].set_title('Chiffres 0 à 9 — jeu de test')
    axes[1].set_title('Pair / impair — jeu de test')
    fig.savefig(sortie / 'confusions.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 6, figsize=(12, 4), constrained_layout=True)
    for ax, indice, vrai, predit in zip(axes.flat, test[:12], y_test[:12], predictions[:12]):
        ax.imshow(chiffres.images[indice], cmap='gray_r')
        ax.set_title(f'Vrai {vrai} / prédit {predit}')
        ax.axis('off')
    fig.savefig(sortie / 'chiffres_test.png', dpi=150)
    plt.close(fig)
    sauver_json(sortie / 'predictions.json', {'indices_test': test.tolist(),
                'vrai': y_test.tolist(), 'predit': predictions.tolist(),
                'probabilites_classes_0_a_9': probabilites.tolist(),
                'vrai_pair': pair_test.tolist(), 'predit_pair': pred_pair.tolist(),
                'probabilite_pair': proba_pair.tolist()})
    sauver_json(sortie / 'partition.json', {'apprentissage': apprentissage.tolist(), 'test': test.tolist()})
    rapport = {'projet': 'classification', 'donnees': 'sklearn.load_digits / UCI Optical Digits',
               'sha256_donnees': hashlib.sha256(chiffres.data.astype('<f8').tobytes() +
                                                chiffres.target.astype('<i8').tobytes()).hexdigest(),
               'n_apprentissage': len(apprentissage), 'n_test': len(test), 'graine': graine,
               'modele': 'StandardScaler + LogisticRegression(C=1, max_iter=2000)',
               'multiclasse': multiclasse, 'binaire': binaire,
               'duree_secondes': time.perf_counter() - debut, 'environnement': environnement()}
    sauver_json(sortie / 'rapport.json', rapport)
    print(f'Classification : {len(test)} images de test ; accuracy={multiclasse["accuracy"]:.4f}', flush=True)
    return rapport


if __name__ == '__main__':
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument('--sortie', type=Path, default=RACINE / 'resultats')
    analyseur.add_argument('--graine', type=int, default=42)
    options = analyseur.parse_args()
    executer(options.sortie, options.graine)
