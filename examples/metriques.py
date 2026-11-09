"""Petites métriques lisibles ; l'AP est déléguée à pycocotools."""
import numpy as np


def box_iou(boite_a, boite_b):
    a, b = np.asarray(boite_a), np.asarray(boite_b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union else 0.0


def apparier(boites_vraies, boites_predites, scores, seuil_score=0.5, seuil_iou=0.5):
    """Appariement glouton un-à-un, par confiance décroissante, pour une image."""
    utilises, ious = set(), []
    faux_positifs = 0
    for indice in np.argsort(-np.asarray(scores), kind='stable'):
        if scores[indice] < seuil_score:
            continue
        candidats = [(box_iou(boites_predites[indice], boite), j)
                     for j, boite in enumerate(boites_vraies) if j not in utilises]
        meilleur, j = max(candidats, default=(0.0, -1))
        if j >= 0 and meilleur >= seuil_iou:
            utilises.add(j)
            ious.append(meilleur)
        else:
            faux_positifs += 1
    return len(utilises), faux_positifs, len(boites_vraies) - len(utilises), ious


def confusion_pixels(vrai, predit):
    """Lignes = vérité (fond/personne), colonnes = prédiction."""
    vrai, predit = np.asarray(vrai, dtype=bool), np.asarray(predit, dtype=bool)
    if vrai.shape != predit.shape:
        raise ValueError('Les masques doivent avoir les mêmes dimensions.')
    return np.bincount(2 * vrai.ravel().astype(int) + predit.ravel(), minlength=4).reshape(2, 2)


def scores_pixels(confusion):
    c = np.asarray(confusion, dtype=np.float64)
    intersection = np.diag(c)
    union = c.sum(axis=0) + c.sum(axis=1) - intersection
    iou = np.divide(intersection, union, out=np.full(2, np.nan), where=union > 0)
    denom = c[1].sum() + c[:, 1].sum()
    return {'pixel_accuracy': float(intersection.sum() / c.sum()) if c.sum() else None,
            'iou_fond': float(iou[0]) if union[0] else None,
            'iou_personne': float(iou[1]) if union[1] else None,
            'miou': float(np.nanmean(iou)) if np.any(union) else None,
            'dice_personne': float(2 * intersection[1] / denom) if denom else None,
            'confusion_pixels': c.astype(int).tolist()}
