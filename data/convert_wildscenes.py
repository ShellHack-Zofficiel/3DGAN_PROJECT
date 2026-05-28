#!/usr/bin/env python3
"""
Script de conversion WildScenes: OBJ/PLY → Nuages de Points (NPY)
Convertit les scènes naturelles du dataset WildScenes
"""

import os
import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm

def obj_to_point_cloud(file_path, num_points=2048, normalize=True):
    """Convertir un fichier 3D en nuage de points"""
    try:
        mesh = trimesh.load(file_path)
        points = mesh.sample(num_points)
        
        if normalize:
            center = np.mean(points, axis=0)
            points -= center
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points /= max_dist
        
        return points.astype(np.float32)
    except Exception as e:
        print(f"❌ Erreur: {file_path}: {e}")
        return None

def convert_wildscenes(raw_dir='data/raw_wildscenes', output_dir='data/WildScenes', 
                       num_points=2048):
    """Convertir tous les fichiers du dataset WildScenes"""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    files = list(raw_path.rglob('*.obj')) + list(raw_path.rglob('*.ply')) + list(raw_path.rglob('*.glb'))
    
    if not files:
        print(f"⚠️  Aucun fichier trouvé dans {raw_dir}")
        return
    
    print(f"\n🔄 Conversion de {len(files)} fichiers WildScenes...")
    converted = 0
    
    for file_path in tqdm(files, desc="Conversion"):
        relative_path = file_path.relative_to(raw_path)
        output_file = output_path / f"wildscenes_{converted:04d}.npy"
        
        point_cloud = obj_to_point_cloud(str(file_path), num_points=num_points)
        
        if point_cloud is not None:
            np.save(str(output_file), point_cloud)
            converted += 1
    
    print(f"\n✅ Conversion terminée: {converted} fichiers")
    print(f"📁 Fichiers sauvegardés dans: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convertir WildScenes → Point Clouds NPY')
    parser.add_argument('--raw-dir', default='data/raw_wildscenes', help='Répertoire des fichiers')
    parser.add_argument('--output-dir', default='data/WildScenes', help='Répertoire de sortie')
    parser.add_argument('--num-points', type=int, default=2048, help='Nombre de points')
    
    args = parser.parse_args()
    convert_wildscenes(args.raw_dir, args.output_dir, args.num_points)
