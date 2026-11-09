"""Lance les trois mini-labs de métriques et régénère leurs figures."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in ('classification_metrics.py', 'detection_metrics.py', 'segmentation_metrics.py'):
    print('\n' + '='*64, script, '='*64, flush=True)
    subprocess.run([sys.executable, str(ROOT / 'examples' / script)], check=True, cwd=ROOT)
print('\nTerminé : ouvrir le dossier figures/ pour visualiser les résultats.')
