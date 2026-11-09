"""Pipeline commun aux projets 2 (boîtes) et 3 (masques), sur Penn-Fudan."""
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn)

from .commun import (RACINE, SHA256_PENN, URL_PENN, ajouter_reference, boite_xywh,
                     choisir_images, empreinte, encoder_masque, environnement,
                     evaluer_coco, lire_image, preparer_penn, sauver_json)
from .metriques import apparier, confusion_pixels, scores_pixels


def illustrer(sortie, image, masques, boites, prediction, seuil, segmentation):
    """Montrer la première image évaluée, sans sélection selon la qualité du résultat."""
    fig, axes = plt.subplots(1, 3 if segmentation else 2, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title('Annotations réelles Penn-Fudan')
    axes[1].imshow(image)
    axes[1].set_title(f'Prédictions, confiance ≥ {seuil}')
    if segmentation:
        instances = np.zeros(masques.shape[1:], dtype=int)
        for numero, masque in enumerate(masques, 1):
            instances[masque] = numero
        axes[0].imshow(np.ma.masked_equal(instances, 0), cmap='tab20', alpha=0.65, vmin=0, vmax=20)
        instances_predites = np.zeros_like(instances)
        # Petits scores peints d'abord ; les meilleurs ont priorité dans cette figure seulement.
        for numero in np.argsort(prediction['scores']):
            if prediction['scores'][numero] >= seuil:
                instances_predites[prediction['masks'][numero, 0] >= 0.5] = int(numero) + 1
        axes[1].imshow(np.ma.masked_equal(instances_predites, 0), cmap='tab20', alpha=0.65, vmin=0, vmax=20)
        axes[2].imshow(instances_predites > 0, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Union sémantique prédite : personne')
    else:
        for ax, rectangles in [(axes[0], boites), (axes[1], prediction['boxes'][prediction['scores'] >= seuil])]:
            for boite in rectangles:
                x, y, w, h = boite_xywh(boite)
                ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=2))
    for ax in axes:
        ax.axis('off')
    fig.savefig(sortie / 'apercu.png', dpi=150)
    plt.close(fig)


def executer(segmentation=False, donnees=RACINE / 'data', sortie=RACINE / 'resultats',
             limite=20, graine=42, seuil=0.5, threads=2):
    if not 0 <= seuil <= 1 or threads < 1:
        raise ValueError('Seuil entre 0 et 1, threads >= 1.')
    debut = time.perf_counter()
    nom = 'segmentation' if segmentation else 'detection'
    sortie = Path(sortie) / nom
    sortie.mkdir(parents=True, exist_ok=True)
    dossier = preparer_penn(donnees)
    images = choisir_images(dossier, limite, graine)
    torch.manual_seed(graine)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    # Cache local au dossier des données ; CPU par défaut sur toutes les machines.
    torch.hub.set_dir(str(Path(donnees) / 'poids'))
    if segmentation:
        poids = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        modele = maskrcnn_resnet50_fpn(weights=poids, box_score_thresh=0.001)
    else:
        poids = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        modele = fasterrcnn_resnet50_fpn(weights=poids, box_score_thresh=0.001)
    modele.eval()
    reference = {'info': {'description': 'Penn-Fudan, masques originaux'},
                 'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'personne'}]}
    predictions, manifeste, ious = [], [], []
    tp = fp = fn = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    with torch.inference_mode():
        for identifiant, chemin in enumerate(images, 1):
            image, masques, boites, masque_path = lire_image(chemin)
            ajouter_reference(reference, identifiant, chemin, image, masques, boites)
            resultat = modele([poids.transforms()(image)])[0]
            # COCO : la classe 1 est « person ». Les autres catégories sont hors tâche.
            garder = resultat['labels'] == 1
            prediction = {cle: valeur[garder].cpu().numpy() for cle, valeur in resultat.items()}
            for j, score in enumerate(prediction['scores']):
                element = {'image_id': identifiant, 'category_id': 1, 'score': float(score)}
                if segmentation:
                    # Ne pas ajouter bbox en mode segm : COCO doit utiliser l'aire du masque.
                    element['segmentation'] = encoder_masque(prediction['masks'][j, 0] >= 0.5)
                else:
                    element['bbox'] = boite_xywh(prediction['boxes'][j])
                predictions.append(element)
            if segmentation:
                retenus = prediction['scores'] >= seuil
                union_predite = (prediction['masks'][retenus, 0] >= 0.5).any(axis=0)
                confusion += confusion_pixels(masques.any(axis=0), union_predite)
            else:
                vrais, faux, manques, ious_image = apparier(boites, prediction['boxes'],
                                                           prediction['scores'], seuil, 0.5)
                tp, fp, fn = tp + vrais, fp + faux, fn + manques
                ious.extend(ious_image)
            manifeste.append({'id': identifiant, 'image': chemin.name,
                               'sha256_image': empreinte(chemin), 'sha256_masque': empreinte(masque_path),
                               'instances': len(masques)})
            if identifiant == 1:
                illustrer(sortie, image, masques, boites, prediction, seuil, segmentation)
            print(f'{nom} : {identifiant}/{len(images)} — {chemin.name}', flush=True)
    # Enregistrer avant COCOeval : loadRes ajoute lui-même des champs aux prédictions.
    sauver_json(sortie / 'annotations_coco.json', reference)
    sauver_json(sortie / 'predictions_coco.json', predictions)
    sauver_json(sortie / 'manifeste.json', manifeste)
    scores, courbes = evaluer_coco(reference, predictions, 'segm' if segmentation else 'bbox')
    sauver_json(sortie / 'precision_rappel.json', courbes)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for iou, precisions in courbes['precision'].items():
        ax.plot(courbes['rappel'], precisions, label=f'IoU {iou}')
    ax.set(xlabel='Rappel', ylabel='Précision interpolée COCO', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    fig.savefig(sortie / 'precision_rappel.png', dpi=150)
    plt.close(fig)
    rapport = {'projet': nom, 'donnees': 'Penn-Fudan', 'url_donnees': URL_PENN,
               'sha256_archive': SHA256_PENN, 'n_images': len(images),
               'n_instances': len(reference['annotations']), 'graine': graine,
               'selection': 'Tirage sans remise, numpy.default_rng, puis ordre alphabétique',
               'modele': type(modele).__name__, 'poids': str(poids), 'url_poids': poids.url,
               'sha256_poids': empreinte(Path(torch.hub.get_dir()) / 'checkpoints' / poids.url.rsplit('/', 1)[1]),
               'appareil': 'cpu', 'threads': threads, 'seuil_confiance': seuil,
               'seuil_minimum_modele_pour_ap': 0.001, 'max_detections_modele': 100,
               'redimensionnement': {'min_size': list(modele.transform.min_size), 'max_size': modele.transform.max_size},
               'coco': scores, 'environnement': environnement()}
    if segmentation:
        rapport['seuil_masque'] = 0.5
        rapport['semantique'] = scores_pixels(confusion)
    else:
        rapport['point_fonctionnement'] = {
            'iou_min': 0.5, 'TP': tp, 'FP': fp, 'FN': fn,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'rappel': tp / (tp + fn) if tp + fn else 0.0,
            'iou_moyen_vrais_positifs': float(np.mean(ious)) if ious else None}
    rapport['duree_secondes'] = time.perf_counter() - debut
    sauver_json(sortie / 'rapport.json', rapport)
    return rapport
