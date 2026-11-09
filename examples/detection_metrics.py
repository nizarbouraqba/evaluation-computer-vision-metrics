"""Calcul reproductible de Box IoU, Precision/Recall et AP/mAP simplifiés.

Important : annotations *illustratives*, pas sorties d'un vrai détecteur.
Le calcul AP utilise une interpolation 101 points inspirée de COCO, mais n'est
pas un remplacement complet de l'API officielle pycocotools (aires, maxDets...).
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage import data

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


def box_iou(a, b):
    """IoU de deux boîtes [xmin, ymin, xmax, ymax] en pixels."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ap_at_iou(gts, preds, category, iou_threshold=0.5):
    """AP 101 points (version didactique), correspondance gloutonne par score."""
    gt_by_image = defaultdict(list)
    for g in gts:
        if g["class"] == category:
            gt_by_image[g["image"]].append(g["box"])
    total_gt = sum(map(len, gt_by_image.values()))
    predictions = sorted((p for p in preds if p["class"] == category),
                         key=lambda p: p["score"], reverse=True)
    if total_gt == 0:
        return float("nan"), [], []
    matched = {img: set() for img in gt_by_image}
    tp, fp = [], []
    for p in predictions:
        boxes = gt_by_image.get(p["image"], [])
        pairs = [(box_iou(p["box"], g), k)
                 for k, g in enumerate(boxes) if k not in matched.get(p["image"], set())]
        best_iou, best_idx = max(pairs, default=(0.0, -1))
        if best_iou >= iou_threshold and best_idx >= 0:
            matched[p["image"]].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not tp:
        return 0.0, np.array([]), np.array([])
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / (tp + fp)
    # Enveloppe de précision (interpolation, 101 niveaux de rappel).
    recall_grid = np.linspace(0, 1, 101)
    p_interp = np.array([np.max(precision[recall >= r]) if np.any(recall >= r)
                         else 0.0 for r in recall_grid])
    return float(p_interp.mean()), recall, precision


def main():
    # Exemple 2 classes : 3 objets annotés sur une photo réelle.
    # Coordonnées approximatives choisies manuellement pour apprendre le calcul.
    gts = [
        {"image": "astronaut", "class": "visage", "box": [168, 54, 295, 184]},
        {"image": "astronaut", "class": "combinaison", "box": [110, 195, 405, 490]},
    ]
    preds = [
        {"image": "astronaut", "class": "visage", "box": [175, 65, 291, 176], "score": .95},
        {"image": "astronaut", "class": "visage", "box": [15, 15, 90, 90], "score": .58},
        {"image": "astronaut", "class": "combinaison", "box": [150, 225, 410, 479], "score": .87},
        {"image": "astronaut", "class": "combinaison", "box": [12, 270, 80, 410], "score": .93},
    ]
    for g in gts:
        p = next(p for p in preds if p["class"] == g["class"])
        print(f"IoU {g['class']}: {box_iou(g['box'], p['box']):.3f}")

    print("\nAP par classe (protocole didactique) :")
    for threshold in (0.5, 0.75):
        vals = []
        for name in ("visage", "combinaison"):
            ap, recall, precision = ap_at_iou(gts, preds, name, threshold)
            vals.append(ap)
            print(f"  {name} AP@{threshold:.2f} = {ap:.3f}")
        print(f"  mAP@{threshold:.2f} = {np.nanmean(vals):.3f}")

    # Moyenne sur 10 seuils IoU : interprétation du mAP@0.50:0.95.
    thresholds = np.arange(0.50, 0.951, 0.05)
    ap_values = [ap_at_iou(gts, preds, name, t)[0]
                 for name in ("visage", "combinaison") for t in thresholds]
    print(f"  mAP@0.50:0.95 (didactique) = {np.nanmean(ap_values):.3f}")

    image = data.astronaut()  # photographie réelle fournie avec scikit-image
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(image)
    for g in gts:
        x1, y1, x2, y2 = g["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor="lime", linewidth=2.5))
        ax.text(x1, y1 - 6, f"Réel : {g['class']}", color="white", fontsize=9,
                bbox={"facecolor": "green", "alpha": .8, "pad": 2})
    for p in preds[:1] + preds[2:3]:
        x1, y1, x2, y2 = p["box"]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor="orange", linestyle="--", linewidth=2.5))
        ax.text(x1, y2 + 12, f"Prédit : {p['class']}", color="black", fontsize=9,
                bbox={"facecolor": "orange", "alpha": .8, "pad": 2})
    ax.set_title("Détection : boîtes illustratives (vert = réel, orange = prédit)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "detection_iou_photo_reelle.png", dpi=160)
    plt.close(fig)
    # Deux courbes PR : à IoU=0,75, la prédiction "combinaison" n'est plus
    # suffisamment bien localisée pour compter comme vrai positif.
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for threshold in (0.5, 0.75):
        ap, recall, precision = ap_at_iou(gts, preds, "combinaison", threshold)
        ax.step(np.r_[0, recall], np.r_[1, precision], where="post",
                label=f"IoU≥{threshold:.2f} — AP={ap:.2f}", linewidth=2)
    ax.set(xlabel="Recall (rappel)", ylabel="Precision (précision)",
           xlim=(0, 1.03), ylim=(0, 1.03),
           title="Precision–Recall : influence du seuil IoU")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "detection_precision_recall.png", dpi=160)
    plt.close(fig)
    print("\nNote : ce sont des boîtes définies pour l'exemple, pas des sorties de YOLO.")


if __name__ == "__main__":
    main()
