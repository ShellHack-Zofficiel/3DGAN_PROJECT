# Guide d'utilisation des datasets

## Structure des données

```
data/
├── ShapeNet/              ← Nuages de points convertis (.npy) - UTILISER POUR L'ENTRAÎNEMENT
│   ├── synthetic_0000.npy
│   ├── synthetic_0001.npy
│   └── ...
├── raw_shapenet/          ← Fichiers .obj bruts (temporaire)
│   └── [Télécharger et mettre les fichiers ShapeNet ici]
├── ModelNet/              ← Nuages de points ModelNet (.npy)
└── raw_modelnet/          ← Fichiers .obj bruts ModelNet
```

## Option 1: Données synthétiques (DÉJÀ FAITES)

✅ **200 nuages de points synthétiques** sont déjà générés dans `data/ShapeNet/`

Prêt à entraîner immédiatement !

```bash
python training/train.py
```

## Option 2: Télécharger ShapeNet

### Étape 1: Accéder à ShapeNet
1. Aller sur https://www.shapenet.org/
2. S'inscrire (gratuit)
3. Télécharger ShapeNet Core v2 (ZIP ~55GB)

### Étape 2: Extraire et organiser
```bash
# Extraire dans data/raw_shapenet/
# La structure doit être:
data/raw_shapenet/
  ├── 02691156/              # Category ID (chair)
  │   ├── 1a0bc3d92903b6f9900fa0bfe957beca/
  │   │   └── model.obj
  │   └── ...
  └── ...
```

### Étape 3: Convertir OBJ → NPY
```bash
python data/convert_shapenet.py \
  --raw-dir data/raw_shapenet \
  --output-dir data/ShapeNet \
  --num-points 2048
```

## Option 3: Télécharger ModelNet

### Télécharger ModelNet40 (1.6GB)
```bash
cd data/raw_modelnet
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
```

### Convertir au format NPY
```bash
python data/convert_modelnet.py \
  --raw-dir data/raw_modelnet \
  --output-dir data/ModelNet
```

## Vérifier les données

### Compter les fichiers
```bash
python -c "
from pathlib import Path
npy_files = list(Path('data/ShapeNet').glob('*.npy'))
print(f'✓ {len(npy_files)} fichiers .npy dans data/ShapeNet/')
print(f'Taille totale: {sum(f.stat().st_size for f in npy_files) / 1024**2:.1f} MB')
"
```

### Visualiser un nuage de points
```bash
python -c "
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pc = np.load('data/ShapeNet/synthetic_0000.npy')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
plt.show()
"
```

## Configuration du DataLoader

Le `PointCloudDataset` cherche automatiquement les fichiers `.npy` dans `data/ShapeNet/`:

```python
from training.utils import PointCloudDataset
from torch.utils.data import DataLoader

dataset = PointCloudDataset('data/ShapeNet', num_points=2048)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Notes importantes

- ✅ Les données synthétiques (200 files) sont prêtes pour la démo
- 📊 Pour un entraînement réel, télécharger ShapeNet (~55GB) ou ModelNet (~1.6GB)
- 🔄 `raw_*/` = fichiers temporaires (peuvent être supprimés après conversion)
- 💾 `*/` = données finales (utilisées par l'entraînement)
- 🎲 Les nuages de points sont normalisés dans [-1, 1]
