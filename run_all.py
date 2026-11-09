"""Lancer les trois mini-projets depuis la racine du dépôt."""
from examples.commun import arguments
from examples.classification_metrics import executer as classification
from examples.pietons import executer as pietons

if __name__ == '__main__':
    options = arguments(__doc__)
    classification(options.sortie, options.graine)
    pietons(segmentation=False, **vars(options))
    pietons(segmentation=True, **vars(options))
