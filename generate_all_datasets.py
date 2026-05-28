#!/usr/bin/env python3
"""
Générateur de données synthétiques pour tous les datasets
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

datasets_info = {
    'HumanArt': ('human_art', 50),
    'WildScenes': ('wildscenes', 60),
    'Orbis': ('orbis', 40),
    'ModelNet': ('modelnet', 50),
}

print("\n🎨 Génération de données synthétiques pour chaque dataset...\n")

for dataset_name, (prefix, count) in datasets_info.items():
    output_dir = Path(f'data/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Génération {dataset_name}: ", end='', flush=True)
    
    for i in tqdm(range(count), desc=dataset_name, leave=False):
        num_points = 2048
        shape_type = i % 5
        
        if shape_type == 0:  # Sphère
            theta = np.random.uniform(0, 2*np.pi, num_points)
            phi = np.arccos(2*np.random.rand(num_points) - 1)
            r = 0.7 + 0.1*np.random.randn(num_points)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
        elif shape_type == 1:  # Tore
            u = np.random.uniform(0, 2*np.pi, num_points)
            v = np.random.uniform(0, 2*np.pi, num_points)
            R, r_tore = 1.0, 0.4
            x = (R + r_tore*np.cos(v)) * np.cos(u)
            y = (R + r_tore*np.cos(v)) * np.sin(u)
            z = r_tore * np.sin(v)
            
        elif shape_type == 2:  # Cube
            x = np.random.uniform(-1, 1, num_points)
            y = np.random.uniform(-1, 1, num_points)
            z = np.random.uniform(-1, 1, num_points)
            
        elif shape_type == 3:  # Cylindre
            theta = np.random.uniform(0, 2*np.pi, num_points)
            z = np.random.uniform(-1, 1, num_points)
            r = 0.5 + 0.1*np.random.randn(num_points)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
        else:  # Cône
            theta = np.random.uniform(0, 2*np.pi, num_points)
            z = np.random.uniform(-1, 1, num_points)
            r = np.abs(z) * 0.5
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        
        points = np.stack([x, y, z], axis=1).astype(np.float32)
        points += np.random.normal(0, 0.02, points.shape).astype(np.float32)
        
        center = np.mean(points, axis=0)
        points -= center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points /= max_dist
        
        output_file = output_dir / f'{prefix}_{i:04d}.npy'
        np.save(str(output_file), points)
    
    print(f"✓")

print("\n✅ Tous les datasets sont prêts!\n")
print("📊 Résumé:")
total_size = 0
for dataset_name, (prefix, count) in datasets_info.items():
    output_dir = Path(f'data/{dataset_name}')
    files = list(output_dir.glob('*.npy'))
    size_mb = sum(f.stat().st_size for f in files) / 1024**2
    total_size += size_mb
    print(f"  ✓ {dataset_name}: {len(files)} fichiers ({size_mb:.1f} MB)")

print(f"\n📈 Total: {total_size:.1f} MB de données")
