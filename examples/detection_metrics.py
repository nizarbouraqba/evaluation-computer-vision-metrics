"""Projet 2 : détecter les piétons réels de Penn-Fudan avec Faster R-CNN."""
from .commun import arguments
from .pietons import executer

if __name__ == '__main__':
    options = arguments(__doc__)
    executer(segmentation=False, **vars(options))
