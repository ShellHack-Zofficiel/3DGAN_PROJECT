# Utiliser 3D GAN sur Google Colab

## 🚀 Lancement rapide

### Option 1: Notebook Colab (Recommandé)
1. Ouvrir: **[Colab_Training.ipynb](https://colab.research.google.com/github/ShellHack-Zofficiel/3DGAN_PROJECT/blob/main/Colab_Training.ipynb)**
2. Cliquer sur "Exécuter tout" ▶️
3. Choisir GPU T4 (Runtime → Change runtime type → GPU)

### Option 2: Setup en une ligne
Exécuter dans une cellule Colab:
```python
!curl -sSL https://raw.githubusercontent.com/ShellHack-Zofficiel/3DGAN_PROJECT/main/setup_colab.py | python
```

Puis lancer l'entraînement:
```python
!python training/train.py
```

## 📊 Étapes du notebook

1. ✅ **Vérifier GPU** - Confirmer la présence du T4
2. ✅ **Cloner le projet** - Depuis GitHub
3. ✅ **Installer dépendances** - PyTorch CUDA + libs
4. ✅ **Vérifier les données** - 400 point clouds prêts
5. ✅ **Configurer** - Charger config.json
6. ✅ **Entraîner** - Lancer l'entraînement (1000 epochs)
7. ✅ **Vérifier résultats** - Afficher checkpoints et modèles
8. ✅ **Générer modèles** - Inférence 3D
9. ✅ **Télécharger** - Archive les résultats

## 💾 Accès à Google Drive (optionnel)

Pour persister les résultats dans Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('/content/3DGAN_Project/checkpoints/checkpoint_epoch_1000.pt',
            '/content/drive/MyDrive/3DGAN_checkpoint.pt')
```

## 📝 Commandes utiles

```python
# Entraîner avec config personnalisée
!python training/train.py --epochs 500 --batch-size 16

# Générer 10 modèles aléatoires
!python inference.py --mode random --num 10

# Générer à partir d'une image
!python main.py --mode image --image image.jpg --num-variations 3

# Convertir les données
!python data/convert_shapenet.py --raw-dir data/raw_shapenet

# Générer plus de données synthétiques
!python generate_all_datasets.py --num-samples 500
```

## 🎯 Ressources GPU Colab

- **GPU T4:** 16 GB VRAM
- **CPU:** 2 cores
- **RAM:** 12.7 GB
- **Stockage:** 108 GB (temporaire)

**Durée max:** 12h continue (puis reset)

## 🐛 Troubleshooting

### "CUDA out of memory"
Réduire batch_size dans `configs/config.json`:
```json
{
  "data": {
    "batch_size": 8
  }
}
```

### "Module not found"
Réinstaller les dépendances:
```python
!pip install -q torch torchvision trimesh open3d numpy scipy matplotlib
```

### GPU non détecté
Vérifier le runtime:
```python
import torch
print(torch.cuda.is_available())  # Doit être True
```

## 📦 Fichiers importants

```
3DGAN_PROJECT/
├── Colab_Training.ipynb    ← Notebook Colab complet
├── setup_colab.py          ← Setup en une ligne
├── training/train.py       ← Entraînement
├── inference.py            ← Génération
├── main.py                 ← Interface principale
├── configs/config.json     ← Configuration
└── data/                   ← 400 point clouds
```

## 🔗 Liens

- **GitHub:** https://github.com/ShellHack-Zofficiel/3DGAN_PROJECT
- **Colab Notebook:** [Ouvrir sur Colab](https://colab.research.google.com/github/ShellHack-Zofficiel/3DGAN_PROJECT/blob/main/Colab_Training.ipynb)
- **Dataset GitHub:** [ShapeNet](https://shapenet.org/), [ModelNet](http://modelnet.cs.princeton.edu/)

---

**Prêt? Ouvrir le notebook sur Colab et cliquer sur ▶️!**
