#!/usr/bin/env python3
"""
Script de conversion ModelNet: OBJ → Nuages de Points (NPY)
"""

import os
import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm

def obj_to_point_cloud(obj_path, num_points=2048, normalize=True):
    """Convertir un fichier OBJ en nuage de points"""
    try:
        mesh = trimesh.load(obj_path)
        points = mesh.sample(num_points)
        
        if normalize:
            center = np.mean(points, axis=0)
            points -= center
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points /= max_dist
        
        return points.astype(np.float32)
    except Exception as e:
        print(f"❌ Erreur: {obj_path}: {e}")
        return None

def convert_modelnet(raw_dir='data/raw_modelnet', output_dir='data/ModelNet', 
                     num_points=2048):
    """Convertir tous les fichiers OBJ de ModelNet"""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    obj_files = list(raw_path.rglob('*.obj'))
    
    if not obj_files:
        print(f"⚠️  Aucun fichier .obj trouvé dans {raw_dir}")
        return
    
    print(f"\n🔄 Conversion de {len(obj_files)} fichiers ModelNet...")
    converted = 0
    
    for obj_file in tqdm(obj_files, desc="Conversion"):
        relative_path = obj_file.relative_to(raw_path)
        output_file = output_path / f"{relative_path.parent.parent.name}_{relative_path.stem}.npy"
        
        point_cloud = obj_to_point_cloud(str(obj_file), num_points=num_points)
        
        if point_cloud is not None:
            np.save(str(output_file), point_cloud)
            converted += 1
    
    print(f"\n✅ Conversion terminée: {converted}/{len(obj_files)} fichiers")
    print(f"📁 Fichiers sauvegardés dans: {output_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convertir ModelNet OBJ → Point Clouds NPY')
    parser.add_argument('--raw-dir', default='data/raw_modelnet', help='Répertoire des .obj')
    parser.add_argument('--output-dir', default='data/ModelNet', help='Répertoire de sortie')
    parser.add_argument('--num-points', type=int, default=2048, help='Nombre de points')
    
    args = parser.parse_args()
    convert_modelnet(args.raw_dir, args.output_dir, args.num_points)
