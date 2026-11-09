# Validation de la livraison — 9 novembre 2026

Les valeurs suivantes proviennent des exécutions locales du notebook et des rapports JSON inclus. Aucun score théorique ou repris d'un modèle publié n'est présenté comme résultat de cette expérience.

## Exécution réellement effectuée

- Windows 11, Python 3.12.14, PyTorch 2.7.1+cpu, TorchVision 0.22.1+cpu, CPU, deux threads pour les réseaux.
- Classification : 1 347 images d'apprentissage, 450 de test, partition stratifiée, graine 42. Les ensembles sont disjoints.
- Détection et segmentation : **20 images identiques, 53 instances annotées**, sélection sans remise, graine 42. **Les réseaux n'ont pas été évalués ici sur les 170 images.**
- L'archive entière a été contrôlée : 170 images, 423 identifiants d'instances, concordant avec les fichiers texte d'annotation. Le nombre historique de 345 sur la page web ne décrit pas toutes les annotations ajoutées dans cette archive.
- Notebook : **8 cellules de code exécutées, zéro erreur**, sorties et figures enregistrées.
- Tests unitaires : **9 réussis** ; `pip check` ne signale aucune dépendance incompatible.
- Les AP et AR de détection/segmentation ont été recalculées à partir des JSON sauvegardés, sans réseau ni modèle, et concordent avec les rapports (tolérance 1e-10). Journal : [recalcul_ap.txt](validation/recalcul_ap.txt).
- Figures de classification, détection et segmentation contrôlées visuellement avant livraison.

## Scores mesurés

Scores sur [0, 1], arrondis ici à six décimales ; les JSON conservent la précision complète.

| Projet | Mesure | Valeur |
|---|---|---:|
| Classification, 450 images de test | Accuracy | 0.977778 |
| Classification | Précision macro | 0.977767 |
| Classification | Rappel macro | 0.977661 |
| Classification | F1 macro | 0.977641 |
| Pair / impair, positif = pair | F1 | 0.911565 |
| Pair / impair | ROC-AUC | 0.970464 |
| Pair / impair | Average Precision | 0.960710 |
| Détection, 20 images | Box AP50 | 0.966408 |
| Détection | Box AP75 | 0.940879 |
| Détection | Box AP50:95 (mAP, une classe) | 0.812053 |
| Détection | Précision à confiance 0,5 / IoU 0,5 | 0.557895 |
| Détection | Rappel à confiance 0,5 / IoU 0,5 | 1.000000 |
| Instances, 20 images | Mask AP50 | 0.938456 |
| Instances | Mask AP75 | 0.889306 |
| Instances | Mask AP50:95 (mAP, une classe) | 0.744388 |
| Sémantique dérivée des instances | Pixel Accuracy | 0.954443 |
| Sémantique | IoU personne | 0.803379 |
| Sémantique | Dice personne | 0.890971 |
| Sémantique | mIoU fond/personne | 0.873701 |

Durées enregistrées : classification 1.5 s, détection 111.1 s, segmentation 128.1 s. Elles englobent le chargement, les figures et, le cas échéant, les téléchargements ; ce ne sont pas des mesures de vitesse du réseau seul.

Rapports : [classification](validation/classification/rapport.json), [détection](validation/detection/rapport.json), [segmentation](validation/segmentation/rapport.json). Liste des images et empreintes : `validation/*/manifeste.json`. Paramètres, versions et SHA-256 des poids sont dans les rapports. Les probabilités de classification et prédictions COCO sont livrées pour vérification.

## Limites

La première illustration, `FudanPed00014.png`, montre aussi une personne visible non annotée dans le masque fourni. Une prédiction sur cette personne compte comme erreur face à cette référence. Les annotations ont été conservées telles quelles ; aucune correction guidée par les prédictions n'a été faite.

Ces résultats décrivent exactement les données, modèles et seuils indiqués, sans intervalle de confiance ni sélection de modèle. Les scores sur 20 images ne sont pas des performances sur tout Penn-Fudan et ne sont pas comparables à un classement COCO. Le modèle sémantique est l'union des instances Mask R-CNN. Le découpage Digits ne garantit pas l'indépendance des scripteurs. L'indépendance entre Penn-Fudan et les données historiques d'entraînement des poids n'a pas été auditée. Linux/macOS et le workflow GitHub Actions n'ont pas été exécutés ici. Les petits jeux des tests sont des contrôles mathématiques, sans lien avec les performances annoncées.

## État GitHub vérifié

Dépôt : [nizarbouraqba/evaluation-computer-vision-metrics](https://github.com/nizarbouraqba/evaluation-computer-vision-metrics).

- Dépôt public accessible en lecture, non archivé, branche `main` accessible.
- Compte connecté identifié : `nizarbouraqba`.
- La recherche du connecteur renvoie `push: true` et `admin: true` : mise à jour possible selon ces permissions déclarées.
- Une vérification supplémentaire via l'API de permissions de collaborateur a renvoyé HTTP 403 « Resource not accessible by integration » ; cet endpoint n'est pas accessible à l'intégration. Aucun essai d'écriture n'a été fait pour tester au-delà des droits déclarés.
- Commit observé : `37d55f38cdf764e10627e347b46a8ac07db6e19a`, « Computer Vision Evaluation Metrics ». Le README distant correspond à l'ancienne version à exemples simulés.
- **Aucun push, commit distant ou changement de dépôt n'a été effectué.** La demande portait sur la livraison du ZIP et la vérification de disponibilité.

## Contenu nettoyé

L'archive livrée contient le nouveau dépôt complet, sans historique `.git`, données téléchargées, modèles ou environnement Python. Aucun nom de fichier ou dossier ne contient `PUBLIER_SUR_GITHUB` ou `publier_dans_github` (contrôle insensible à la casse). Les anciennes figures et démonstrations synthétiques sont remplacées. Le dossier original dans Downloads est conservé.

Pour mettre à jour une copie Git existante, remplacer ses anciens fichiers de projet par le contenu de ce dossier, tout en conservant son dossier `.git`, puis examiner `git status` et `git diff`. Ne pas simplement superposer les dossiers en conservant les anciens fichiers obsolètes. Le ZIP est une livraison propre, pas une opération de mise à jour automatique du dépôt distant.
