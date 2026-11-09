# Sources et conditions d'utilisation

Consultées le 9 novembre 2026. Ne pas confondre la licence du code d'une bibliothèque avec les droits de ses images ou de ses poids.

## Chiffres

- [UCI — Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin et C. Kaynak (1998), DOI [10.24432/C50P49](https://doi.org/10.24432/C50P49). La page UCI indique CC BY 4.0 ; conserver l'attribution.
- [Documentation scikit-learn load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) : le sous-ensemble intégré contient 1 797 images 8 × 8 et correspond au test historique UCI. Le découpage 75/25 utilisé ici est propre au mini-projet.
- [Métriques scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).

## Piétons

- [Penn-Fudan — source des données](https://www.cis.upenn.edu/~jshi/ped_html/), 170 images. La page indique 345 piétons ; l'archive vérifiée en contient 423 (masques et annotations texte concordants). Son `readme.txt` et `added-object-list.txt` documentent des annotations supplémentaires. Voir `validation/donnees_controle.json`.
- Liming Wang, Jianbo Shi, Gang Song, I-fan Shen, *Object Detection Combining Recognition and Segmentation*, ACCV 2007 (référence indiquée sur la page du jeu).
- [Tutoriel officiel TorchVision sur Penn-Fudan](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html). Notre protocole utilise des modèles préentraînés gelés, et ne reproduit pas le fine-tuning du tutoriel.
- La page de téléchargement ne fournit pas de licence normalisée explicite. Le `readme.txt` de l'archive réserve les droits aux auteurs ou ayants droit et précise que la republication peut nécessiter leur autorisation. Les photographies originales et l'archive ne sont pas redistribuées ici. Les aperçus annotés et extraits d'annotations servent à documenter cette expérience ; cela ne leur attribue pas une licence ouverte. Conserver l'attribution et examiner ces conditions avant une autre redistribution.

## Modèles et évaluateur

- [Faster R-CNN ResNet-50 FPN](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html), poids explicites `COCO_V1`.
- [Mask R-CNN ResNet-50 FPN](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html), poids explicites `COCO_V1`.
- [Versions compatibles PyTorch / TorchVision](https://pytorch.org/get-started/previous-versions/).
- [COCOeval — code de référence](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) et [paquet pycocotools, prise en charge de Windows](https://pypi.org/project/pycocotools/2.0.10/).
- [Conditions des modèles préentraînés TorchVision](https://docs.pytorch.org/vision/0.22/models.html) : les conditions des données d'entraînement peuvent aussi s'appliquer. Les poids ne sont pas redistribués ici.

Les valeurs de performance publiées dans ces documentations ne sont pas reprises comme résultats du projet. Seuls les rapports produits par nos exécutions alimentent `VALIDATION.md`.
