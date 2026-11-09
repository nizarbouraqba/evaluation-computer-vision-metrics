"""Chargement de Penn-Fudan, traçabilité et évaluation COCO officielle."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

RACINE = Path(__file__).resolve().parents[1]
URL_PENN = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
# Empreinte calculée sur l'archive officielle téléchargée le 9 novembre 2026.
SHA256_PENN = '9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2'


def empreinte(chemin):
    calcul = hashlib.sha256()
    with Path(chemin).open('rb') as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b''):
            calcul.update(bloc)
    return calcul.hexdigest()


def sauver_json(chemin, contenu):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(contenu, indent=2, ensure_ascii=False,
                                  allow_nan=False), encoding='utf-8')


def environnement():
    paquets = ['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'Pillow',
               'torch', 'torchvision', 'pycocotools']
    return {'python': platform.python_version(), 'systeme': platform.platform(),
            'versions': {p: importlib.metadata.version(p) for p in paquets},
            'date_utc': datetime.now(timezone.utc).isoformat()}


def preparer_penn(dossier):
    """Télécharger une archive vérifiée ; ne jamais remplacer des données par du simulé."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    archive = dossier / 'PennFudanPed.zip'
    if not archive.exists():
        temporaire = archive.with_suffix('.part')
        print('Téléchargement de Penn-Fudan (~51 Mo)...', flush=True)
        with urllib.request.urlopen(URL_PENN, timeout=120) as source:
            with temporaire.open('wb') as destination:
                while bloc := source.read(1024 * 1024):
                    destination.write(bloc)
        temporaire.replace(archive)
    if empreinte(archive) != SHA256_PENN:
        raise ValueError('Archive Penn-Fudan différente de la version vérifiée : '
                         'retirer data/PennFudanPed.zip et retélécharger depuis la source.')
    # Réextraire les PNG depuis l'archive contrôlée, même si le cache a été modifié.
    with zipfile.ZipFile(archive) as contenu:
        base = dossier.resolve()
        for membre in contenu.infolist():
            cible = (base / membre.filename).resolve()
            if not cible.is_relative_to(base):
                raise ValueError('Chemin interdit dans l’archive.')
        contenu.extractall(base)
    return dossier / 'PennFudanPed'


def choisir_images(dossier, limite=20, graine=42):
    chemins = sorted((Path(dossier) / 'PNGImages').glob('*.png'))
    if len(chemins) != 170:
        raise ValueError(f'170 images attendues, {len(chemins)} trouvées.')
    if not 0 <= limite <= len(chemins):
        raise ValueError('--limite doit être entre 0 (tout) et 170.')
    # Tirage sans remise sur les deux sites ; ordre stable pour les rapports.
    if limite:
        indices = np.random.default_rng(graine).choice(len(chemins), limite, replace=False)
        chemins = [chemins[i] for i in sorted(indices)]
    return chemins


def lire_image(chemin):
    image = Image.open(chemin).convert('RGB')
    masque_path = chemin.parent.parent / 'PedMasks' / (chemin.stem + '_mask.png')
    with Image.open(masque_path) as fichier:
        etiquettes = np.asarray(fichier).copy()
    identifiants = np.unique(etiquettes)
    masques = etiquettes[None, :, :] == identifiants[identifiants != 0, None, None]
    boites = []
    for masque in masques:
        y, x = np.where(masque)
        # Coordonnées continues [xmin, ymin, xmax, ymax[ : dernier pixel inclus via +1.
        boites.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
    return image, masques, np.asarray(boites, dtype=np.float32), masque_path


def encoder_masque(masque):
    from pycocotools import mask as coco_mask
    rle = coco_mask.encode(np.asfortranarray(masque, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def boite_xywh(boite):
    x1, y1, x2, y2 = map(float, boite)
    return [x1, y1, x2 - x1, y2 - y1]


def ajouter_reference(coco, identifiant, chemin, image, masques, boites):
    coco['images'].append({'id': identifiant, 'file_name': chemin.name,
                           'width': image.width, 'height': image.height})
    for masque, boite in zip(masques, boites):
        coco['annotations'].append({
            'id': len(coco['annotations']) + 1, 'image_id': identifiant,
            'category_id': 1, 'iscrowd': 0, 'bbox': boite_xywh(boite),
            'area': int(masque.sum()), 'segmentation': encoder_masque(masque)})


def evaluer_coco(reference, predictions, type_iou):
    """AP et AR calculées par pycocotools, y compris s'il n'y a aucune prédiction."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    vrai = COCO()
    vrai.dataset = reference
    vrai.createIndex()
    if predictions:
        predit = vrai.loadRes(predictions)
    else:
        predit = COCO()
        predit.dataset = {**reference, 'annotations': []}
        predit.createIndex()
    evaluation = COCOeval(vrai, predit, type_iou)
    evaluation.params.catIds = [1]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    noms = ['AP_50_95', 'AP_50', 'AP_75', 'AP_petit', 'AP_moyen', 'AP_grand',
            'AR_max1', 'AR_max10', 'AR_max100', 'AR_petit', 'AR_moyen', 'AR_grand']
    # COCO renvoie -1 lorsqu'une catégorie de taille ne possède aucune référence.
    scores = {nom: float(valeur) if valeur >= 0 else None
              for nom, valeur in zip(noms, evaluation.stats)}
    courbes = {}
    for indice, seuil in [(0, '0.50'), (5, '0.75')]:
        courbes[seuil] = evaluation.eval['precision'][indice, :, 0, 0, 2].tolist()
    return scores, {'rappel': evaluation.params.recThrs.tolist(), 'precision': courbes}


def arguments(description):
    analyseur = argparse.ArgumentParser(description=description)
    analyseur.add_argument('--donnees', type=Path, default=RACINE / 'data')
    analyseur.add_argument('--sortie', type=Path, default=RACINE / 'resultats')
    analyseur.add_argument('--limite', type=int, default=20, help='0 = les 170 images')
    analyseur.add_argument('--graine', type=int, default=42)
    analyseur.add_argument('--seuil', type=float, default=0.5,
                          help='Seuil de confiance pour P/R, figures et union sémantique')
    analyseur.add_argument('--threads', type=int, default=2, help='Nombre de threads CPU')
    options = analyseur.parse_args()
    if not 0 <= options.seuil <= 1 or options.threads < 1:
        analyseur.error('Seuil entre 0 et 1 ; threads >= 1.')
    return options
