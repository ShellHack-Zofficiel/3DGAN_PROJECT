#!/usr/bin/env python3
"""
Quick Inference Script - Générer rapidement des modèles 3D
"""

import torch
import os
from main import ThreeDGANInterface


def quick_generate(num_models=5, format='obj', output_dir='./generated_models'):
    """
    Générer rapidement des modèles 3D
    
    Args:
        num_models: nombre de modèles à générer
        format: 'obj' ou 'fbx'
        output_dir: dossier de sortie
    """
    print("\n" + "="*60)
    print("🎨 3D GAN - Générateur Rapide")
    print("="*60)
    
    # Vérifier GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    
    # Initialiser l'interface
    print("✓ Initialisation du modèle...")
    gan = ThreeDGANInterface(
        config_path='configs/config.json',
        device=device
    )
    
    gan.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer les modèles
    print(f"\n✓ Génération de {num_models} modèles...")
    models = gan.generate_random(num_samples=num_models)
    
    # Exporter
    print(f"\n✓ Export en {format.upper()}...")
    paths = gan.export_obj(models) if format == 'obj' else gan.export_fbx(models)
    
    print("\n" + "="*60)
    print(f"✅ Succès! {len(paths)} modèle(s) générés")
    print(f"📁 Localisation: {os.path.abspath(output_dir)}")
    print("="*60 + "\n")
    
    return paths


def quick_image_to_3d(image_path, num_variations=3, format='obj', output_dir='./generated_models'):
    """
    Générer rapidement un modèle 3D à partir d'une image
    
    Args:
        image_path: chemin vers l'image
        num_variations: nombre de variations
        format: 'obj' ou 'fbx'
        output_dir: dossier de sortie
    """
    if not os.path.exists(image_path):
        print(f"❌ Erreur: Image non trouvée: {image_path}")
        return None
    
    print("\n" + "="*60)
    print("🖼️  3D GAN - Image vers 3D")
    print("="*60)
    
    # Vérifier GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    
    # Initialiser l'interface
    print("✓ Initialisation du modèle...")
    gan = ThreeDGANInterface(
        config_path='configs/config.json',
        device=device
    )
    
    gan.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Générer à partir de l'image
    print(f"\n✓ Génération de {num_variations} modèle(s) à partir de: {image_path}")
    models = gan.image_to_3d(image_path, num_samples=num_variations)
    
    # Exporter
    print(f"\n✓ Export en {format.upper()}...")
    paths = gan.export_obj(models) if format == 'obj' else gan.export_fbx(models)
    
    print("\n" + "="*60)
    print(f"✅ Succès! {len(paths)} modèle(s) générés à partir de l'image")
    print(f"📁 Localisation: {os.path.abspath(output_dir)}")
    print("="*60 + "\n")
    
    return paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick 3D Generation Script')
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'image'],
                        help='random: générer des modèles aléatoires, image: image-to-3D')
    parser.add_argument('--num', type=int, default=5,
                        help='Nombre de modèles (mode random) ou variations (mode image)')
    parser.add_argument('--image', type=str,
                        help='Chemin vers l\'image (mode image)')
    parser.add_argument('--format', type=str, default='obj',
                        choices=['obj', 'fbx'],
                        help='Format d\'export')
    parser.add_argument('--output', type=str, default='./generated_models',
                        help='Dossier de sortie')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'random':
            quick_generate(
                num_models=args.num,
                format=args.format,
                output_dir=args.output
            )
        elif args.mode == 'image':
            if not args.image:
                print("❌ Erreur: --image est requis pour le mode 'image'")
            else:
                quick_image_to_3d(
                    image_path=args.image,
                    num_variations=args.num,
                    format=args.format,
                    output_dir=args.output
                )
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
