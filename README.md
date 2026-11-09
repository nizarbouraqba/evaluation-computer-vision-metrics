# Évaluation de modèles de vision — trois mini-projets réels

**Master Data & IA — classification, détection, segmentation sémantique et d'instances.**

Les images et annotations viennent de jeux de données réels. Les prédictions sont produites par des modèles exécutés par le code : aucune étiquette, boîte, masque ou performance de démonstration n'est inventé.

| Mini-projet | Données | Modèle | Mesures |
|---|---|---|---|
| 1. Reconnaître des chiffres manuscrits | UCI Optical Digits, version `load_digits` | Standardisation + régression logistique entraînée localement | Accuracy, précision/rappel/F1 macro, rapport par classe, confusion ; pair/impair : F1, ROC-AUC, Average Precision |
| 2. Détecter des piétons | Penn-Fudan, photographies et masques d'origine | Faster R-CNN ResNet-50 FPN, poids `COCO_V1` | Box IoU, précision/rappel à seuil fixe, AP50, AP75, AP50:95, AR |
| 3. Segmenter des piétons | Même sélection Penn-Fudan | Mask R-CNN ResNet-50 FPN, poids `COCO_V1` | Mask AP50/AP75/AP50:95 ; union des instances : Pixel Accuracy, IoU, Dice, mIoU |

Le projet 3 couvre deux niveaux : conserver un masque par personne (instances), puis réunir les masques en une classe « personne » (sémantique binaire). Il n'entraîne pas un réseau sémantique indépendant. Les projets 2 et 3 partagent le chargement et l'évaluateur pour garder le code court et cohérent.

## Installation

Utiliser **Python 3.11 ou 3.12, 64 bits**. La configuration vérifiée et les résultats sont dans [VALIDATION.md](VALIDATION.md). Pas de GPU requis : le protocole utilise le CPU et deux threads. Prévoir quelques Go libres pour l'environnement, les modèles, les données et les résultats ; les temps réels de cette machine figurent dans les rapports. L'accès Internet est requis au premier lancement.

Depuis la racine `Evaluation_CV_GitHub`, sous Windows PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
```

Si `py` n'existe pas, remplacer la première commande par `python -m venv .venv`, après vérification de `python --version`. Aucune activation PowerShell n'est nécessaire. Python 3.14 n'est pas la cible de ces versions figées.

Linux (CPU) :

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run_all.py
```

Sur macOS, installer directement `requirements.txt` depuis PyPI, sans l'index CPU Linux/Windows. macOS et Linux ne sont pas présentés comme testés localement.

`requirements.txt` fixe les dépendances directes. `requirements-dev.txt` ajoute les tests et Jupyter. `validation/environnement-pip.txt` fige aussi les dépendances transitives de la machine de validation Windows ; il décrit cet environnement, sans garantir une installation identique sur un autre système. PyTorch et TorchVision doivent rester dans la paire compatible indiquée.

| Dépendance | Rôle |
|---|---|
| NumPy, SciPy, scikit-learn | tableaux, apprentissage et métriques de classification |
| PyTorch, TorchVision | inférence des modèles préentraînés et transformations |
| Pillow, Matplotlib | lecture des PNG et figures |
| pycocotools | calcul de référence COCO pour les boîtes et masques |
| pytest, nbformat, nbclient, ipykernel, notebook | validation et utilisation du notebook (développement) |

## Exécution et notebook

Toutes les commandes suivantes supposent le Python de l'environnement actif ; sous Windows sans activation, remplacer `python` par `.\.venv\Scripts\python.exe`.

```bash
# Chaque mini-projet peut être lancé séparément.
python -m examples.classification_metrics
python -m examples.detection_metrics --limite 20 --graine 42
python -m examples.segmentation_metrics --limite 20 --graine 42

# Tout le jeu Penn-Fudan ; garder les résultats dans un dossier distinct.
python run_all.py --limite 0 --sortie resultats_complets

# Tests et interface Jupyter.
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m examples.recalculer_ap --resultats validation
python -m notebook notebooks/TP_Evaluation_Computer_Vision.ipynb
```

Le [notebook](notebooks/TP_Evaluation_Computer_Vision.ipynb) contient trois parcours en français, exécute le même code que les scripts, affiche les résultats et propose des questions d'interprétation. Les paramètres `LIMITE`, `GRAINE`, `DONNEES` et `SORTIE` sont dans la première cellule. Redémarrer le noyau puis **Exécuter tout** pour reproduire l'expérience. Les sorties enregistrées correspondent à la validation, pas à une exécution fictive.

Par défaut : 20 images tirées sans remise avec `numpy.default_rng(42)` dans la liste alphabétique des 170 images. C'est un **sous-ensemble de démonstration**, pas un score sur l'ensemble de Penn-Fudan. La liste exacte est exportée. `--limite 0` utilise les 170 images. Le même tirage sert à la détection et à la segmentation. Une nouvelle exécution dans le même dossier remplace ses rapports : utiliser `--sortie` pour comparer les expériences.

## Données, poids et traçabilité

### Chiffres manuscrits

`sklearn.datasets.load_digits()` fournit 1 797 vraies images 8 × 8, valeurs de pixels de 0 à 16. Elles sont incluses dans scikit-learn : aucun téléchargement séparé. Ce sous-ensemble correspond au fichier de test historique UCI, que **nous redécoupons** en apprentissage et test ; nous ne prétendons donc pas reproduire le protocole officiel UCI.

Le découpage est stratifié par chiffre, 75 % apprentissage / 25 % test (1 347 / 450), graine 42. Le `StandardScaler` est ajusté dans une `Pipeline` exclusivement sur l'apprentissage. Les deux modèles sont entraînés séparément : chiffre 0–9 et pair/impair. Hyperparamètres fixés à l'avance, aucun réglage sur le test. Les scripteurs ne sont pas séparés par ce découpage : il ne mesure pas une généralisation garantie à des personnes inédites. La parité est calculée à partir de l'étiquette réelle du chiffre ; ce n'est pas une annotation simulée.

### Piétons Penn-Fudan

Le script télécharge l'[archive officielle](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip) dans `data/` (53 723 336 octets lors de la vérification). Elle contient **170 images et 423 instances**, comptées indépendamment dans les masques et les fichiers texte d'annotation. La page web annonce encore 345 piétons ; le `readme.txt` de l'archive explique l'ajout d'annotations de petites personnes et de personnes occultées, répertoriées dans `added-object-list.txt`. Nous utilisons toutes les instances des images sélectionnées, sans en supprimer pour retrouver le chiffre historique. Les PNG `PedMasks` fournissent un identifiant par instance, 0 pour le fond. Les rectangles sont les enveloppes des pixels annotés ; la borne maximale est exclusive (`max + 1`). Les masques restent à leur résolution originale pour l'évaluation.

SHA-256 de l'archive utilisée :

```text
9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2
```

L'archive est vérifiée à chaque lancement ; les PNG sont réextraits depuis cette version contrôlée. Une erreur de téléchargement ou d'empreinte arrête l'expérience. Il n'existe aucun remplacement automatique par des données fictives.

Les poids explicites `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` et `MaskRCNN_ResNet50_FPN_Weights.COCO_V1` sont téléchargés depuis `download.pytorch.org` (~160 et ~170 Mio) et mis en cache dans `data/poids/`. TorchVision contrôle leur empreinte au téléchargement. Les rapports conservent ensuite leur SHA-256 complet, l'URL et le nom exact. Aucun poids ni archive de données n'est inclus dans le ZIP ou destiné à Git.

Les modèles sont gelés (`eval`, `inference_mode`), sans entraînement ni adaptation sur Penn-Fudan. Seule la catégorie COCO `person`, identifiant 1, est évaluée. Les transformations des poids convertissent en tenseur ; les modèles appliquent leur normalisation et redimensionnement internes (petit côté 800, grand côté au maximum 1 333), puis rendent les prédictions à la taille initiale. Les détails effectifs sont enregistrés. Les poids proviennent d'un entraînement COCO ; aucune vérification exhaustive de recoupement des images entre leurs données d'entraînement et Penn-Fudan n'a été faite. Il s'agit d'une évaluation pédagogique de transfert, pas d'une garantie d'indépendance des corpus.

## Comprendre les métriques

- Classification : `Accuracy = bonnes prédictions / total`. La précision mesure la proportion de positifs prédits corrects ; le rappel, la proportion de vrais positifs retrouvés. `F1 = 2PR/(P+R)`. Macro = moyenne non pondérée sur les 10 chiffres. Pour pair/impair, la classe positive est « pair », seuil 0,5 ; ROC-AUC et Average Precision utilisent les probabilités, pas les décisions binaires. Cette Average Precision scikit-learn n'est pas l'AP COCO à 101 points.
- Détection : `IoU = intersection / union` des boîtes continues. Pour le point de fonctionnement, prédictions triées par confiance, seuil de confiance 0,5, IoU ≥ 0,5, appariement un-à-un. Une deuxième détection du même piéton devient un faux positif. L'IoU moyen affiché ne porte que sur les vrais positifs et ne mesure pas les objets manqués.
- AP COCO : calcul par `pycocotools.COCOeval`, 101 niveaux de rappel, IoU de 0,50 à 0,95 par pas de 0,05, au plus 100 détections par image. AP50 et AP75 fixent l'IoU. AR100 est le rappel moyen sur les seuils IoU, avec 100 détections maximum. Une seule classe évaluée : la mAP sur les classes se confond avec l'AP « personne ». Les courbes fournies sont les précisions interpolées COCO.
- Pour l'AP, on conserve les prédictions jusqu'au seuil interne de 0,001 ; **on n'applique pas le seuil 0,5** réservé aux figures, au point précision/rappel et à l'union sémantique. Le modèle applique toujours sa suppression des doublons (NMS) et sa limite de 100 détections, toutes catégories avant filtrage « personne ». C'est le protocole de cette expérience, pas une soumission au benchmark COCO.
- Segmentation d'instances : chaque masque prédit est binarisé à 0,5. `COCOeval(iouType='segm')` compare chaque instance à une annotation réelle. Mask AP distingue les personnes même lorsqu'elles partagent une classe.
- Sémantique : union des masques vrais ; union des masques prédits de confiance ≥ 0,5. `Pixel Accuracy = pixels corrects / tous les pixels`, `IoU = TP/(TP+FP+FN)`, `Dice = 2TP/(2TP+FP+FN)`. La confusion est **cumulée sur tous les pixels** des images avant calcul, puis `mIoU = moyenne(IoU fond, IoU personne)`. Ce n'est pas la moyenne des scores image par image. Une classe absente de la vérité et des prédictions est exclue de la mIoU et notée `null`.

Tous les scores sont sur **[0, 1]**. Une catégorie de taille COCO sans référence est notée `null` (au lieu du -1 de COCO). Précision/rappel au point fixe : dénominateur nul → 0. Aucune prédiction avec des références présentes → AP = 0. La Pixel Accuracy peut masquer de mauvais contours quand le fond domine : lire aussi Dice, IoU et Mask AP.

Les annotations restent une limite réelle : dans `FudanPed00014.png`, la première image montrée, une deuxième personne visible n'a pas d'instance dans le masque fourni. Sa détection est donc pénalisée par l'évaluation. Nous conservons les annotations originales, sans les compléter après avoir vu les prédictions.

## Fichiers et résultats

```text
Evaluation_CV_GitHub/
├── README.md, VALIDATION.md, SOURCES.md
├── requirements.txt, requirements-dev.txt, run_all.py
├── examples/
│   ├── classification_metrics.py
│   ├── detection_metrics.py
│   ├── segmentation_metrics.py
│   ├── pietons.py              # inférence commune et figures
│   ├── commun.py               # données, empreintes et COCO
│   └── metriques.py            # IoU, appariement et pixels
├── notebooks/TP_Evaluation_Computer_Vision.ipynb
├── tests/test_metrics.py
├── validation/                 # résultats réellement exécutés livrés
├── .github/workflows/tests.yml
└── .gitignore
```

Les nouvelles exécutions écrivent dans `resultats/` (ignoré par Git). Chaque projet produit un `rapport.json` et des figures. La classification conserve aussi les indices de partition, étiquettes et probabilités. Détection/segmentation : `manifeste.json` avec images et empreintes, annotations et prédictions COCO, courbes précision–rappel. Ces fichiers permettent de recalculer l'AP sans relancer le réseau. Les rapports contiennent versions, paramètres, date et durée ; les durées peuvent inclure les premiers téléchargements et ne sont pas une mesure de débit d'inférence.

Voir [VALIDATION.md](VALIDATION.md) pour les scores mesurés, le périmètre exécuté et les limites. Les petites fixtures des tests servent uniquement à vérifier les calculs et les cas limites ; elles ne constituent aucun exemple de performance du projet. Le workflow CI teste ces métriques et exécute la classification réelle ; il ne prétend pas exécuter les deux réseaux ni le notebook complet à chaque push.

## Sources, droits et suite du projet

Les références primaires, auteurs et conditions sont dans [SOURCES.md](SOURCES.md). Les données et poids conservent leurs conditions propres ; aucune licence générale n'est attribuée aux photographies Penn-Fudan. Le dossier livré remplace les anciens exemples, leurs figures et le notebook. Les documents de publication de l'ancienne version sont exclus de cette livraison. L'état d'accès au dépôt GitHub est consigné dans `VALIDATION.md` ; cette livraison locale ne publie rien automatiquement.
