#!/usr/bin/env python3
"""
Générateur de données synthétiques pour le test/démo
Crée des nuages de points synthétiques dans data/ShapeNet/
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_synthetic_point_clouds(output_dir='data/ShapeNet', 
                                   num_samples=200, 
                                   num_points=2048):
    """
    Générer des nuages de points synthétiques pour la démo/test
    
    Args:
        output_dir: répertoire de sortie
        num_samples: nombre de nuages à générer
        num_points: nombre de points par nuage
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Génération de {num_samples} nuages de points synthétiques...")
    
    for i in tqdm(range(num_samples), desc="Génération"):
        # Variations de formes: sphère, cube, tore, etc.
        shape_type = i % 5
        
        if shape_type == 0:
            # Sphère avec bruit
            theta = np.random.uniform(0, 2*np.pi, num_points)
            phi = np.arccos(2*np.random.rand(num_points) - 1)
            r = 0.7 + 0.1*np.random.randn(num_points)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
        elif shape_type == 1:
            # Cube avec bruit
            x = np.random.uniform(-1, 1, num_points)
            y = np.random.uniform(-1, 1, num_points)
            z = np.random.uniform(-1, 1, num_points)
            
        elif shape_type == 2:
            # Cylindre
            theta = np.random.uniform(0, 2*np.pi, num_points)
            z = np.random.uniform(-1, 1, num_points)
            r = 0.5 + 0.1*np.random.randn(num_points)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
        elif shape_type == 3:
            # Tore
            u = np.random.uniform(0, 2*np.pi, num_points)
            v = np.random.uniform(0, 2*np.pi, num_points)
            R, r = 1.0, 0.4
            x = (R + r*np.cos(v)) * np.cos(u)
            y = (R + r*np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            
        else:
            # Cône
            theta = np.random.uniform(0, 2*np.pi, num_points)
            z = np.random.uniform(-1, 1, num_points)
            r = np.abs(z) * 0.5
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        
        # Ajouter du bruit
        points += np.random.normal(0, 0.02, points.shape).astype(np.float32)
        
        # Normaliser
        center = np.mean(points, axis=0)
        points -= center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points /= max_dist
        
        # Sauvegarder
        output_file = output_path / f'synthetic_{i:04d}.npy'
        np.save(str(output_file), points)
    
    print(f"✅ {num_samples} fichiers générés dans {output_dir}/")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Générer des nuages de points synthétiques'
    )
    parser.add_argument('--output-dir', default='data/ShapeNet',
                        help='Répertoire de sortie')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Nombre de nuages à générer')
    parser.add_argument('--num-points', type=int, default=2048,
                        help='Nombre de points par nuage')
    
    args = parser.parse_args()
    generate_synthetic_point_clouds(args.output_dir, args.num_samples, args.num_points)
