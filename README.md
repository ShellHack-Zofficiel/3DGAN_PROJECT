# 3DGAN_Project - Générateur de Modèles 3D par IA

## 🎯 Objectif

Créer une IA génératrice de modèles 3D réalistes basée sur un **GAN (Generative Adversarial Network)**.

### Fonctionnalités principales:
- ✅ Génération aléatoire de modèles 3D réalistes
- ✅ **Image-to-3D**: Donner une image et obtenir un modèle 3D
- ✅ Reconstruction de maillages (Poisson, Ball Pivoting)
- ✅ Export en **OBJ** et **FBX** (pour Blender, Unity, Unreal Engine)
- ✅ Entraînement sur ShapeNet/ModelNet

---

## 📦 Installation

### Prérequis
- Python 3.8+
- CUDA 11.0+ (optionnel, pour GPU)

### Étapes d'installation

```bash
# 1. Cloner le projet
cd 3DGAN_Project

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install torch>=2.0 torchvision>=0.15
pip install numpy scipy matplotlib
pip install open3d>=0.17
pip install trimesh>=3.22
pip install Pillow

# 4. (Optionnel) Installer Kaolin pour optimisations
pip install kaolin

# 5. (Optionnel) Installer Blender avec API Python
# Télécharger depuis https://www.blender.org/download/
# et activer l'API Python
```

---

## 🚀 Utilisation

### 1. Générer des modèles 3D aléatoires

```bash
# Générer 5 modèles aléatoires en OBJ
python main.py --mode random --num-models 5 --format obj

# Générer en FBX (nécessite Blender)
python main.py --mode random --num-models 5 --format fbx

# Avec checkpoint pré-entraîné
python main.py --mode random --num-models 5 --checkpoint checkpoints/model.pt
```

**Exemple Python:**
```python
from main import ThreeDGANInterface

# Initialiser l'interface
gan = ThreeDGANInterface(config_path='configs/config.json')

# Générer 10 modèles
models = gan.generate_random(num_samples=10)

# Exporter en OBJ
gan.export_obj(models)
```

---

### 2. Générer un modèle 3D à partir d'une image

```bash
# Donner une image et obtenir un modèle 3D
python main.py --mode image --image path/to/image.jpg --num-variations 3 --format obj

# Exemple:
python main.py --mode image --image data/chair.jpg --num-variations 5 --format fbx
```

**Exemple Python:**
```python
from main import ThreeDGANInterface

gan = ThreeDGANInterface(config_path='configs/config.json')

# Générer à partir d'une image
models = gan.image_to_3d('path/to/image.jpg', num_samples=3)

# Exporter en OBJ
gan.export_obj(models)
```

---

### 3. Entraîner le modèle

```bash
python training/train.py
```

**Préparer les données:**
1. Télécharger ShapeNet ou ModelNet
2. Placer les fichiers dans `data/`
3. Convertir en format `.npy` ou `.pt`

---

## 📁 Structure du projet

```
3DGAN_Project/
├── configs/
│   └── config.json              # Hyperparamètres
├── data/                        # Datasets (ShapeNet, ModelNet)
├── models/
│   ├── generator.py             # Architectures du générateur
│   ├── discriminator.py         # Architectures du discriminateur
│   └── losses.py                # Fonctions de perte
├── training/
│   ├── train.py                 # Script d'entraînement
│   └── utils.py                 # Utilitaires
├── export/
│   ├── to_obj.py                # Export OBJ
│   └── to_fbx.py                # Export FBX
├── generated_models/            # Modèles générés
├── checkpoints/                 # Checkpoints sauvegardés
├── main.py                      # Point d'entrée principal
└── README.md
```

---

## ⚙️ Configuration

Modifier `configs/config.json` pour ajuster:

```json
{
  "model": {
    "latent_dim": 128,          # Dimension du vecteur latent
    "num_points": 2048,          # Nombre de points 3D
    "generator": {
      "layers": [128, 256, 512, 1024]  # Couches du générateur
    }
  },
  "training": {
    "epochs": 200,
    "learning_rate_g": 0.0002,
    "learning_rate_d": 0.0002,
    "batch_size": 32
  },
  "image_to_3d": {
    "enabled": true,             # Activer image-to-3D
    "encoder_backbone": "resnet50"
  }
}
```

---

## 🔧 Exemples avancés

### Générer et reconstruire un maillage

```python
from main import ThreeDGANInterface

gan = ThreeDGANInterface()

# Générer modèles
models = gan.generate_random(num_samples=5)

# Reconstruire les maillages avec Poisson
meshes = gan.reconstruct_mesh(models, method='poisson')

# Exporter en FBX
gan.export_fbx(models, method='poisson')
```

### Générer avec paramètres personnalisés

```python
# Générer avec une dimension latente différente
models = gan.generate_random(
    num_samples=10,
    latent_dim=256  # Dimension personnalisée
)

# Avec scaling
gan.export_obj(models, scale=2.0)
```

### Traitement batch

```python
# Générer et exporter plusieurs modèles d'une image
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']

for image in images:
    gan.image_to_3d_export(
        image_path=image,
        num_variations=5,
        export_format='fbx'
    )
```

---

## 📊 Datasets disponibles

### ShapeNet
- 3 millions de modèles 3D
- Télécharger: https://shapenet.org/
- Format: OBJ, PLY, DAE

### ModelNet
- 151,128 modèles 3D
- Télécharger: http://modelnet.cs.princeton.edu/
- Catégories: 10, 40

---

## 🎮 Export pour moteurs 3D

### Blender
```bash
python main.py --mode random --num-models 5 --format fbx

# Ouvrir dans Blender:
# File > Import > Autodesk FBX (.fbx)
```

### Unity
1. Exporter en FBX
2. Importer dans Assets/
3. Configurer les materials

### Unreal Engine
1. Exporter en FBX
2. Importer via Content Browser
3. Configurer les matériaux

---

## 📈 Entraînement

### Préparer les données

```python
from training.utils import PointCloudDataset

# Créer un dataset
dataset = PointCloudDataset(
    data_dir='data/',
    num_points=2048
)

# DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)
```

### Lancer l'entraînement

```bash
python training/train.py

# Avec configuration personnalisée
python training/train.py --config configs/config.json --epochs 300
```

---

## 🐛 Troubleshooting

### Erreur: "Blender not found"
```bash
# Installer Blender ou définir le chemin
python main.py --blender-path "C:\Program Files\Blender\blender.exe"
```

### Erreur: "CUDA out of memory"
- Réduire `batch_size` dans config.json
- Réduire `num_points`
- Utiliser `--device cpu`

### Erreur: "open3d import failed"
```bash
pip install open3d
```

---

## 📚 Ressources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Open3D Documentation](http://www.open3d.org/)
- [Trimesh Documentation](https://trimesh.org/)
- [GAN Research Papers](https://arxiv.org/list/cs.LG)

---

## 📝 Licence

MIT License - Libre d'utilisation

---

## 👨‍💻 Auteur

3DGAN_Project - Générateur de Modèles 3D par Intelligence Artificielle

Pour toute question ou contribution, n'hésitez pas à créer une issue ou un PR!

---

## 🎓 Apprentissage

### Étapes pour maîtriser le projet:

1. **Comprendre les GANs**
   - Lire: "Generative Adversarial Networks" (Goodfellow et al.)
   - Comprendre: Générateur vs Discriminateur

2. **Données 3D**
   - Maîtriser les nuages de points
   - Comprendre les maillages

3. **Entraînement**
   - Créer un dataset
   - Entraîner le modèle
   - Optimiser les hyperparamètres

4. **Export et intégration**
   - Exporter en différents formats
   - Intégrer dans des moteurs 3D

---

Bonne chance dans votre exploration de la génération 3D! 🚀
