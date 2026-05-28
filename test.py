#!/usr/bin/env python3
"""
Test Script - Vérifier que tout fonctionne correctement
"""

import torch
import os
import sys
from pathlib import Path


def test_imports():
    """Tester l'import des modules"""
    print("\n" + "="*60)
    print("🧪 Test 1: Vérification des imports")
    print("="*60)
    
    try:
        from models.generator import Generator, ConditionalGenerator
        print("✓ Generator et ConditionalGenerator")
        
        from models.discriminator import Discriminator, PointNetDiscriminator
        print("✓ Discriminator et PointNetDiscriminator")
        
        from models.losses import ChamferDistance, GANLoss, CombinedLoss
        print("✓ Loss functions")
        
        from training.utils import PointCloudDataset, EarlyStopping
        print("✓ Training utils")
        
        from export.to_obj import PointCloudToOBJ
        print("✓ OBJ export")
        
        from export.to_fbx import FBXExporter
        print("✓ FBX export")
        
        from training.train import Trainer
        print("✓ Trainer")
        
        from main import ThreeDGANInterface
        print("✓ Main interface")
        
        print("\n✅ Tous les imports sont OK!")
        return True
    except ImportError as e:
        print(f"\n❌ Erreur d'import: {e}")
        return False


def test_config():
    """Tester la configuration"""
    print("\n" + "="*60)
    print("🧪 Test 2: Vérification de la configuration")
    print("="*60)
    
    try:
        from training.utils import load_config
        config = load_config('configs/config.json')
        
        print(f"✓ Config loaded")
        print(f"  - Project: {config['project']['name']}")
        print(f"  - Latent dim: {config['model']['latent_dim']}")
        print(f"  - Num points: {config['model']['num_points']}")
        print(f"  - Batch size: {config['data']['batch_size']}")
        
        print("\n✅ Configuration OK!")
        return True
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False


def test_models():
    """Tester l'initialisation des modèles"""
    print("\n" + "="*60)
    print("🧪 Test 3: Initialisation des modèles")
    print("="*60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Device: {device}")
        
        from models.generator import Generator
        from models.discriminator import PointNetDiscriminator
        
        # Generator
        gen = Generator(latent_dim=128, num_points=2048).to(device)
        print("✓ Generator créé")
        
        # Test forward
        z = torch.randn(2, 128).to(device)
        out = gen(z)
        assert out.shape == (2, 2048, 3), f"Expected (2, 2048, 3), got {out.shape}"
        print(f"✓ Generator forward pass: {out.shape}")
        
        # Discriminator
        disc = PointNetDiscriminator(num_points=2048).to(device)
        print("✓ Discriminator créé")
        
        # Test forward
        validity = disc(out)
        assert validity.shape == (2, 1), f"Expected (2, 1), got {validity.shape}"
        print(f"✓ Discriminator forward pass: {validity.shape}")
        
        print("\n✅ Modèles OK!")
        return True
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interface():
    """Tester l'interface principale"""
    print("\n" + "="*60)
    print("🧪 Test 4: Interface principale")
    print("="*60)
    
    try:
        from main import ThreeDGANInterface
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        interface = ThreeDGANInterface(
            config_path='configs/config.json',
            device=device
        )
        print("✓ Interface créée")
        
        # Test generation
        models = interface.generate_random(num_samples=2)
        assert models.shape == (2, 2048, 3), f"Expected (2, 2048, 3), got {models.shape}"
        print(f"✓ Random generation: {models.shape}")
        
        print("\n✅ Interface OK!")
        return True
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structure():
    """Vérifier la structure du projet"""
    print("\n" + "="*60)
    print("🧪 Test 5: Structure du projet")
    print("="*60)
    
    required_files = [
        'configs/config.json',
        'models/generator.py',
        'models/discriminator.py',
        'models/losses.py',
        'training/train.py',
        'training/utils.py',
        'export/to_obj.py',
        'export/to_fbx.py',
        'main.py',
        'README.md',
        'requirements.txt',
    ]
    
    required_dirs = [
        'data',
        'models',
        'training',
        'export',
        'configs',
    ]
    
    all_ok = True
    
    # Check files
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"❌ MISSING: {file}")
            all_ok = False
    
    # Check directories
    for dir in required_dirs:
        path = Path(dir)
        if path.is_dir():
            print(f"✓ {dir}/")
        else:
            print(f"❌ MISSING: {dir}/")
            all_ok = False
    
    if all_ok:
        print("\n✅ Structure OK!")
    else:
        print("\n❌ Certains fichiers manquent!")
    
    return all_ok


def main():
    """Lancer tous les tests"""
    print("\n" + "="*70)
    print(" "*15 + "🧪 3DGAN PROJECT TEST SUITE 🧪")
    print("="*70)
    
    results = []
    
    # Test 1
    results.append(("Imports", test_imports()))
    
    # Test 2
    results.append(("Configuration", test_config()))
    
    # Test 3
    results.append(("Modèles", test_models()))
    
    # Test 4
    results.append(("Interface", test_interface()))
    
    # Test 5
    results.append(("Structure", test_structure()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 RÉSUMÉ")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{passed}/{total} tests passés")
    
    if passed == total:
        print("\n" + "="*70)
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("="*70)
        print("\nVous pouvez maintenant utiliser le projet:")
        print("  - python main.py --mode random --num-models 5")
        print("  - python inference.py --mode random --num 5")
        print("  - python main.py --mode image --image path/to/image.jpg")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
