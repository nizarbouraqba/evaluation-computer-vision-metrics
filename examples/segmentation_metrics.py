"""Métriques de segmentation sémantique ET segmentation d'instances.

Pour expliquer les métriques sans entraînement, les masques sont créés à la main.
La photographie d'astronaute provient de scikit-image ; ce ne sont PAS
les prédictions d'un véritable modèle. Exécuter depuis la racine du dépôt :
    python examples/segmentation_metrics.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


def binary_metrics(gt, pred):
    """Renvoie (IoU, Dice). Convention : deux masques vides ont le score 1."""
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    iou = intersection / union if union else 1.0
    denom = gt.sum() + pred.sum()
    dice = 2 * intersection / denom if denom else 1.0
    return float(iou), float(dice)


def miou_multiclasse(gt, pred, classes):
    """mIoU, en ignorant les classes absentes à la fois du vrai et du prédit."""
    values = {}
    for cls in classes:
        gt_c, pred_c = gt == cls, pred == cls
        union = np.logical_or(gt_c, pred_c).sum()
        if union > 0:
            values[cls] = float(np.logical_and(gt_c, pred_c).sum() / union)
    return (float(np.mean(list(values.values()))) if values else float('nan')), values


def instance_ap_at_iou(gt_masks, predictions, threshold=0.5):
    """AP des masques d'une seule classe dans UNE image (version didactique).

    gt_masks : liste de matrices booléennes, une par objet réel.
    predictions : liste de dictionnaires {'mask': matrice bool, 'score': float}.
    Note : ne remplace pas la métrique COCO officielle (plusieurs images, classes,
    règles crowd/ignore/area/maxDets et agrégation complète).
    """
    if not gt_masks:
        return float('nan'), np.array([]), np.array([])
    order = sorted(predictions, key=lambda p: p['score'], reverse=True)
    used, true_pos, false_pos = set(), [], []
    for pred in order:
        candidates = [(binary_metrics(pred['mask'], gt)[0], i)
                      for i, gt in enumerate(gt_masks) if i not in used]
        best_iou, best_id = max(candidates, default=(0.0, -1))
        success = best_id >= 0 and best_iou >= threshold
        if success:
            used.add(best_id)
        true_pos.append(int(success))
        false_pos.append(int(not success))
    if not order:
        return 0.0, np.array([]), np.array([])
    tp = np.cumsum(true_pos)
    fp = np.cumsum(false_pos)
    recall = tp / len(gt_masks)
    precision = tp / (tp + fp)
    grid = np.linspace(0, 1, 101)
    interpolated = [np.max(precision[recall >= r]) if np.any(recall >= r)
                    else 0.0 for r in grid]
    return float(np.mean(interpolated)), recall, precision


def semantic_demo():
    # 3x3, intersection=3, union=5 : résultat facile à vérifier à la main.
    a = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    b = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
    iou, dice = binary_metrics(a, b)
    assert np.isclose(iou, .6) and np.isclose(dice, .75)
    print(f"Masques binaires 3x3 : IoU={iou:.3f}, Dice={dice:.3f}")

    # Photo réelle ; masques 100 % fictifs superposés pour la démonstration.
    photo = data.astronaut()
    h, w = photo.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    gt = np.zeros((h, w), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[((xx - 240)/70)**2 + ((yy - 110)/76)**2 <= 1] = 1
    pred[((xx - 250)/74)**2 + ((yy - 120)/73)**2 <= 1] = 1
    gt[((xx - 275)/160)**2 + ((yy - 365)/145)**2 <= 1] = 2
    pred[((xx - 282)/158)**2 + ((yy - 355)/142)**2 <= 1] = 2
    pixel_acc = float((gt == pred).mean())
    miou, per_class = miou_multiclasse(gt, pred, classes=[0,1,2])
    print(f"Pixel Accuracy={pixel_acc:.3f} ; mIoU={miou:.3f}")
    print("IoU par classe (fond, zone A, zone B) :",
          {k: round(v, 3) for k, v in per_class.items()})

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(photo)
    axes[0].set_title("Photographie réelle")
    for ax, mask, title in zip(axes[1:], [gt, pred],
                               ["Masque de référence simulé", "Masque prédit simulé"]):
        ax.imshow(photo, alpha=.45)
        ax.imshow(np.ma.masked_equal(mask, 0), cmap="viridis", alpha=.65, vmin=0, vmax=2)
        ax.set_title(title)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f"Segmentation pédagogique : Pixel Accuracy={pixel_acc:.2f} ; mIoU={miou:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / "segmentation_masques_photo_reelle.png", dpi=160)
    plt.close(fig)


def instances_demo():
    # Deux objets distincts de la MÊME classe. Image synthétique pour éviter
    # de confondre ces masques avec des annotations humaines sur une vraie photo.
    h, w = 170, 260
    y, x = np.mgrid[:h, :w]
    circle = lambda cx, cy, r: (x-cx)**2 + (y-cy)**2 <= r*r
    gt_masks = [circle(75,85,36), circle(182,85,29)]
    preds = [
        {"mask": circle(80,85,36), "score": .96},
        {"mask": circle(195,85,29), "score": .80},
        {"mask": circle(125,20,12), "score": .40},
    ]
    print("\nSegmentation d'instances — AP des masques (didactique) :")
    for threshold in (0.50, 0.75):
        ap, _, _ = instance_ap_at_iou(gt_masks, preds, threshold)
        print(f"  Mask AP@{threshold:.2f} = {ap:.3f}")
    aps = [instance_ap_at_iou(gt_masks, preds, t)[0]
           for t in np.arange(0.5, 0.951, 0.05)]
    print(f"  Mask AP@0.50:0.95 (une classe) = {np.mean(aps):.3f}")

    gt_img = np.zeros((h,w), dtype=np.uint8)
    pr_img = np.zeros((h,w), dtype=np.uint8)
    for i, mask in enumerate(gt_masks, start=1):
        gt_img[mask] = i
    for i, pred in enumerate(preds, start=1):
        pr_img[pred['mask']] = i
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    axes[0].imshow(gt_img, cmap='tab10', vmin=0, vmax=9)
    axes[1].imshow(pr_img, cmap='tab10', vmin=0, vmax=9)
    axes[0].set_title('Instances réelles simulées : 2 objets')
    axes[1].set_title('Instances prédites simulées : 3 objets')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'segmentation_instances_mask_ap.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    semantic_demo()
    instances_demo()
    print("\nFigures enregistrées dans :", OUT)
    print("Note : données et prédictions illustratives, aucune IA entraînée dans ce TP.")
