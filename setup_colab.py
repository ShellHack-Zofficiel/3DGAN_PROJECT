#!/usr/bin/env python3
"""
Script de setup rapide pour Colab
Exécuter dans une cellule Colab: !curl -sSL https://raw.githubusercontent.com/ShellHack-Zofficiel/3DGAN_PROJECT/main/setup_colab.py | python
"""

import os
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    """Exécuter une commande shell"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"❌ Erreur: {result.stderr}")
    return result.returncode == 0

print("=" * 70)
print("🚀 SETUP 3D GAN POUR COLAB")
print("=" * 70)

# 1. Cloner le repo
print("\n1️⃣ Clonage du projet...")
project_dir = Path('/content/3DGAN_Project')
if not project_dir.exists():
    run_cmd(f'git clone https://github.com/ShellHack-Zofficiel/3DGAN_PROJECT.git {project_dir}')
    print("✓ Projet cloné")
else:
    print("✓ Projet déjà présent")

os.chdir(project_dir)

# 2. Installer PyTorch CUDA
print("\n2️⃣ Installation PyTorch...")
run_cmd('pip install -q torch>=2.0 torchvision>=0.15 --index-url https://download.pytorch.org/whl/cu121')
print("✓ PyTorch installé")

# 3. Installer les dépendances
print("\n3️⃣ Installation des dépendances...")
run_cmd('pip install -q numpy scipy matplotlib Pillow trimesh open3d scikit-learn tqdm pyyaml')
print("✓ Dépendances installées")

# 4. Vérifier GPU
print("\n4️⃣ Vérification GPU...")
import torch
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# 5. Vérifier les données
print("\n5️⃣ Vérification des données...")
data_files = sum(1 for _ in Path('data').rglob('*.npy'))
print(f"✓ {data_files} fichiers de données trouvés")

print("\n" + "=" * 70)
print("✅ SETUP TERMINÉ - Prêt pour l'entraînement!")
print("=" * 70)
print("\n📖 Utilisation:")
print("  python training/train.py              # Entraîner le modèle")
print("  python inference.py --mode random     # Générer des modèles")
print("  python main.py --mode image --image <path>  # Image vers 3D")
