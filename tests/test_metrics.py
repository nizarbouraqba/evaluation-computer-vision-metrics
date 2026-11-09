"""Tests de validation des métriques — exécutables avec python -m pytest -q."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from detection_metrics import box_iou, ap_at_iou
from segmentation_metrics import binary_metrics, miou_multiclasse, instance_ap_at_iou


def test_box_iou():
    assert np.isclose(box_iou([0,0,10,10], [5,0,15,10]), 1/3)
    assert box_iou([0,0,10,10], [11,11,21,21]) == 0
    assert box_iou([0,0,10,10], [0,0,10,10]) == 1


def test_binary_segmentation():
    a=np.array([[0,1,1],[0,1,1],[0,0,0]])
    b=np.array([[0,1,1],[0,0,1],[0,1,0]])
    assert np.allclose(binary_metrics(a,b), (.6,.75))
    miou, per_cls=miou_multiclasse(a,b,[0,1])
    assert len(per_cls)==2 and 0 <= miou <= 1


def test_detection_ap_perfect():
    gt=[{'image':'A','class':'chat','box':[0,0,10,10]}]
    pr=[{'image':'A','class':'chat','box':[0,0,10,10],'score':.9}]
    ap,rec,prec=ap_at_iou(gt,pr,'chat',.5)
    assert ap == 1 and rec.tolist()==[1] and prec.tolist()==[1]


def test_instance_ap_perfect():
    gt=[np.array([[1,1],[0,0]],bool)]
    pr=[{'mask':np.array([[1,1],[0,0]],bool),'score':.9}]
    ap,rec,prec=instance_ap_at_iou(gt,pr,.5)
    assert ap == 1 and rec.tolist()==[1] and prec.tolist()==[1]


def test_instance_ap_bad_mask():
    gt=[np.array([[1,1],[0,0]],bool)]
    pr=[{'mask':np.array([[0,0],[1,1]],bool),'score':.9}]
    assert instance_ap_at_iou(gt,pr,.5)[0] == 0
