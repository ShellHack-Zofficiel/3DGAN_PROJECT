#!/usr/bin/env python3
"""
Script de conversion ShapeNet: OBJ → Nuages de Points (NPY)
Convertit les fichiers .obj bruts en nuages de points normalisés
"""

import os
import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm

def obj_to_point_cloud(obj_path, num_points=2048, normalize=True):
    """
    Convertir un fichier OBJ en nuage de points
    
    Args:
        obj_path: chemin vers le fichier OBJ
        num_points: nombre de points à échantillonner
        normalize: normaliser dans [-1, 1]
    
    Returns:
        point_cloud: array (num_points, 3)
    """
    try:
        mesh = trimesh.load(obj_path)
        
        # Échantillonner les points à partir de la surface du maillage
        points = mesh.sample(num_points)
        
        if normalize:
            # Normaliser dans [-1, 1]
            center = np.mean(points, axis=0)
            points -= center
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points /= max_dist
        
        return points.astype(np.float32)
    
    except Exception as e:
        print(f"❌ Erreur conversion {obj_path}: {e}")
        return None

def convert_shapenet(raw_dir='data/raw_shapenet', output_dir='data/ShapeNet', 
                     num_points=2048):
    """
    Convertir tous les fichiers OBJ de ShapeNet en nuages de points NPY
    
    Args:
        raw_dir: répertoire des fichiers .obj bruts
        output_dir: répertoire de sortie pour les fichiers .npy
        num_points: nombre de points par nuage
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Trouver tous les fichiers .obj
    obj_files = list(raw_path.rglob('*.obj'))
    
    if not obj_files:
        print(f"⚠️  Aucun fichier .obj trouvé dans {raw_dir}")
        return
    
    print(f"\n🔄 Conversion de {len(obj_files)} fichiers OBJ...")
    converted = 0
    
    for obj_file in tqdm(obj_files, desc="Conversion"):
        # Générer un nom de sortie unique
        relative_path = obj_file.relative_to(raw_path)
        output_file = output_path / f"{relative_path.parent.name}_{relative_path.stem}.npy"
        
        # Convertir
        point_cloud = obj_to_point_cloud(str(obj_file), num_points=num_points)
        
        if point_cloud is not None:
            np.save(str(output_file), point_cloud)
            converted += 1
    
    print(f"\n✅ Conversion terminée: {converted}/{len(obj_files)} fichiers")
    print(f"📁 Fichiers sauvegardés dans: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convertir ShapeNet OBJ → Point Clouds NPY'
    )
    parser.add_argument('--raw-dir', default='data/raw_shapenet',
                        help='Répertoire des fichiers .obj bruts')
    parser.add_argument('--output-dir', default='data/ShapeNet',
                        help='Répertoire de sortie des fichiers .npy')
    parser.add_argument('--num-points', type=int, default=2048,
                        help='Nombre de points par nuage')
    
    args = parser.parse_args()
    convert_shapenet(args.raw_dir, args.output_dir, args.num_points)
