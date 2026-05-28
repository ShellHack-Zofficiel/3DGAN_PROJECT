# 🚀 Guide de Démarrage Rapide - 3DGAN_Project

## ⚡ 30 secondes pour commencer

### 1. Installation des dépendances (2 min)
```bash
pip install -r requirements.txt
```

### 2. Tester l'installation (1 min)
```bash
python test.py
```

### 3. Générer votre premier modèle 3D (1 min)
```bash
python inference.py --mode random --num 3 --format obj
```

✅ **Fait!** Vos modèles sont dans `generated_models/`

---

## 📖 Cas d'usage courants

### 🎲 Générer des modèles aléatoires

```bash
# 5 modèles OBJ
python inference.py --mode random --num 5 --format obj

# 10 modèles FBX (nécessite Blender)
python inference.py --mode random --num 10 --format fbx

# Avec checkpoint personnalisé
python main.py --mode random --num-models 5 --checkpoint checkpoints/my_model.pt
```

### 🖼️ Image vers Modèle 3D

```bash
# Une image → 3 modèles 3D différents
python inference.py --mode image --image photo.jpg --num 3 --format obj

# Personnalisé
python main.py --mode image --image furniture.png --num-variations 5 --format fbx
```

---

## 💻 Utilisation en Python

### Générer et exporter rapidement

```python
from main import ThreeDGANInterface

# Initialiser
gan = ThreeDGANInterface()

# Générer 5 modèles aléatoires
models = gan.generate_random(num_samples=5)

# Exporter en OBJ
gan.export_obj(models)

# Ou en FBX avec maillage
gan.export_fbx(models, method='poisson')
```

### Image vers 3D en Python

```python
gan = ThreeDGANInterface()

# Charger une image et générer
models = gan.image_to_3d('my_image.jpg', num_samples=3)

# Exporter
gan.export_obj(models, names=['model_1.obj', 'model_2.obj', 'model_3.obj'])
```

---

## 📂 Où sont mes fichiers?

```
generated_models/
├── model_20240527_123045_000.obj
├── model_20240527_123045_001.obj
└── model_20240527_123045_002.obj
```

---

## ⚙️ Configuration

### Modifier les paramètres

Éditer `configs/config.json`:

```json
{
  "model": {
    "latent_dim": 128,        // Plus grand = plus de variation
    "num_points": 2048         // Plus = plus détails (plus lourd)
  },
  "inference": {
    "device": "cuda"           // "cuda" ou "cpu"
  }
}
```

---

## 🎮 Utiliser vos modèles dans Blender

1. **Générer les modèles**
   ```bash
   python inference.py --mode random --num 5 --format fbx
   ```

2. **Ouvrir Blender**
   - File → Import → Autodesk FBX
   - Sélectionner les fichiers dans `generated_models/`

3. **Personnaliser**
   - Ajouter des matériaux
   - Ajuster la lumière
   - Exporter pour votre jeu

---

## 🔧 Dépannage rapide

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `CUDA out of memory` | `python inference.py --device cpu` |
| `Blender not found` | Installer Blender ou utiliser OBJ au lieu de FBX |
| Lent sur CPU | Installer CUDA ou réduire `num_points` |

---

## 📚 Prochaines étapes

### 1. Comprendre le projet
```bash
cat README.md
```

### 2. Explorer les options
```bash
python inference.py --help
python main.py --help
```

### 3. Utiliser votre propre dataset
```
data/
├── shapenet_models/
│   ├── model_1.npy
│   ├── model_2.npy
│   └── ...
└── modelnet_models/
    └── ...
```

### 4. Entraîner le modèle
```bash
python training/train.py
```

---

## 💡 Astuce Pro

### Générer en batch avec Python

```python
from main import ThreeDGANInterface

gan = ThreeDGANInterface()

# Générer 100 modèles pour votre jeu
for i in range(0, 100, 10):
    models = gan.generate_random(num_samples=10)
    gan.export_obj(models)
    print(f"Progress: {i+10}/100")
```

### Combiner image et aléatoire

```python
# Générer des variations d'une image
base_models = gan.image_to_3d('object.jpg', num_samples=3)
gan.export_fbx(base_models)

# Générer aussi des modèles aléatoires
random_models = gan.generate_random(num_samples=5)
gan.export_fbx(random_models)
```

---

## ✨ Que faire avec vos modèles?

- 🎮 **Jeux vidéo** - Importer dans Unity/Unreal
- 🖼️ **Visualisation 3D** - Blender, ThreeJS
- 🏭 **Impression 3D** - Exporter en STL
- 🤖 **IA** - Utilisez comme données d'entraînement
- 💼 **Prototypage** - Design rapide

---

## 🎓 Ressources utiles

- **PyTorch**: https://pytorch.org/tutorials/
- **3D Format Guide**: https://en.wikipedia.org/wiki/Wavefront_.obj_file
- **Blender Python API**: https://docs.blender.org/api/current/

---

## ❓ Questions fréquentes

**Q: Puis-je utiliser les modèles commercialement?**  
R: Oui! Licence MIT - libre d'utilisation.

**Q: Quel GPU recommandez-vous?**  
R: RTX 3060+ pour entraînement, tout GPU pour inférence.

**Q: Où obtenir des datasets?**  
R: ShapeNet, ModelNet, Thingiverse, Sketchfab

**Q: Combien de temps pour entraîner?**  
R: GPU: 1-2 jours, CPU: 1-2 semaines

---

## 🚀 Vous êtes prêt!

```bash
# Lancez votre première génération
python inference.py --mode random --num 5 --format obj
```

Bonne chance! 🎉
