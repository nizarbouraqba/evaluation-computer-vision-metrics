"""Projet 3 : segmenter les piétons réels avec Mask R-CNN (instances + union sémantique)."""
from .commun import arguments
from .pietons import executer

if __name__ == '__main__':
    options = arguments(__doc__)
    executer(segmentation=True, **vars(options))
