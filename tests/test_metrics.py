"""Cas limites mathématiques ; aucune performance de projet ne vient de ces fixtures."""
import numpy as np
import pytest
from examples.commun import encoder_masque, evaluer_coco
from examples.metriques import apparier, box_iou, confusion_pixels, scores_pixels


def test_iou_continu_et_contact():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0


def test_doublon_ne_compte_pas_deux_fois():
    tp, fp, fn, _ = apparier([[0, 0, 10, 10]], [[0, 0, 10, 10]] * 2, [0.8, 0.9])
    assert (tp, fp, fn) == (1, 1, 0)


def test_confiance_et_absence_predictions():
    assert apparier([[0, 0, 10, 10]], [], [])[:3] == (0, 0, 1)
    assert apparier([[0, 0, 10, 10]], [[0, 0, 10, 10]], [0.49])[:3] == (0, 0, 1)


def test_pixels_avec_erreurs():
    vrai = np.array([[1, 1], [0, 0]])
    predit = np.array([[1, 0], [1, 0]])
    mesures = scores_pixels(confusion_pixels(vrai, predit))
    assert mesures['pixel_accuracy'] == 0.5
    assert mesures['iou_personne'] == pytest.approx(1 / 3)
    assert mesures['dice_personne'] == 0.5


def test_classe_absente_exclue_de_miou():
    mesures = scores_pixels(confusion_pixels(np.zeros((2, 2)), np.zeros((2, 2))))
    assert mesures['miou'] == 1
    assert mesures['iou_personne'] is None
    assert mesures['dice_personne'] is None


@pytest.mark.parametrize('type_iou', ['bbox', 'segm'])
@pytest.mark.parametrize('vide', [False, True])
def test_coco_parfait_ou_sans_prediction(type_iou, vide):
    masque = np.ones((8, 8), dtype=bool)
    reference = {'info': {}, 'images': [{'id': 1, 'width': 8, 'height': 8}],
                 'categories': [{'id': 1, 'name': 'personne'}],
                 'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                  'bbox': [0, 0, 8, 8], 'area': 64, 'iscrowd': 0,
                                  'segmentation': encoder_masque(masque)}]}
    prediction = {'image_id': 1, 'category_id': 1, 'score': 0.9}
    prediction.update({'bbox': [0, 0, 8, 8]} if type_iou == 'bbox'
                      else {'segmentation': encoder_masque(masque)})
    scores, _ = evaluer_coco(reference, [] if vide else [prediction], type_iou)
    assert scores['AP_50_95'] == pytest.approx(0 if vide else 1)
    assert scores['AP_grand'] is None
