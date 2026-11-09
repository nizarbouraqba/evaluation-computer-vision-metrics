"""Recalculer les AP sauvegardées sans télécharger de données ni charger les modèles."""
import argparse
import json
from pathlib import Path
from .commun import RACINE, evaluer_coco


def verifier(dossier):
    for tache, type_iou in [('detection', 'bbox'), ('segmentation', 'segm')]:
        chemin = Path(dossier) / tache
        lire = lambda nom: json.loads((chemin / nom).read_text(encoding='utf-8'))
        rapport = lire('rapport.json')
        scores, _ = evaluer_coco(lire('annotations_coco.json'),
                                 lire('predictions_coco.json'), type_iou)
        for nom, valeur in scores.items():
            ancienne = rapport['coco'][nom]
            if valeur is None or ancienne is None:
                assert valeur is ancienne, f'{tache} : {nom} différent'
            else:
                assert abs(valeur - ancienne) < 1e-10, f'{tache} : {nom} différent'
        print(f'{tache} : toutes les AP et AR reproduites depuis les prédictions sauvegardées.')


if __name__ == '__main__':
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument('--resultats', type=Path, default=RACINE / 'validation')
    verifier(analyseur.parse_args().resultats)
