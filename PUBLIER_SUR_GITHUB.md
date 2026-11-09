# Publier ce projet sur votre compte GitHub (Windows)

Le projet est prêt à être publié. **Ne téléversez pas le fichier ZIP comme un unique fichier GitHub** : il faut publier le contenu du dossier pour que le code et README.md soient visibles directement.

1. Ouvrir https://github.com/new et créer un dépôt, par exemple **evaluation-computer-vision-metrics**, de préférence Public pour que l'enseignant puisse l'ouvrir. Ne pas cocher « Add a README », puisqu'il est déjà fourni.
2. Décompresser le ZIP et ouvrir PowerShell *dans le dossier contenant `README.md`*.
3. Exécuter les commandes suivantes en remplaçant `VOTRE_LOGIN` par votre identifiant GitHub :

```powershell
git init -b main
git add .
git commit -m "Add reproducible computer vision evaluation examples"
git remote add origin https://github.com/VOTRE_LOGIN/evaluation-computer-vision-metrics.git
git push -u origin main
```

4. Copier le lien de votre dépôt depuis GitHub pour la remise.

Si `git` n'est pas installé, utiliser [GitHub Desktop](https://desktop.github.com/) : File → Add local repository → Create a repository here → Publish repository.

**Vérifier sur GitHub** : `README.md`, `examples/`, `figures/`, `requirements.txt`, `tests/` et `run_all.py` doivent être visibles. Pour démontrer que les scripts fonctionnent localement, faire :

```powershell
python -m pip install -r requirements.txt
python run_all.py
python -m pytest -q
```

Le formulaire impose aussi deux liens **Google Drive** (PPTX et vidéo <= 7 minutes) ; le dépôt GitHub ne remplace pas ces liens.
