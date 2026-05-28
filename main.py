#!/usr/bin/env python3
"""
3DGAN Project - Point d'entrée principal
Réseau antagoniste génératif pour la génération de modèles 3D

Fonctionnalités :
    - Générer des modèles 3D aléatoires à partir de vecteurs latents
    - Image-to-3D : générer des modèles 3D à partir d'images
    - Export en formats OBJ et FBX
    - Inférence en temps réel
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import argparse
import json
from pathlib import Path
from PIL import Image
from datetime import datetime

# Import project modules
from models.generator import Generator, ConditionalGenerator
from models.discriminator import Discriminator, PointNetDiscriminator
from training.utils import load_config, load_checkpoint
from export.to_obj import PointCloudToOBJ, PointCloudToMesh, OBJExporter
from export.to_fbx import FBXExporter, UniversalExporter


class ThreeDGANInterface:
    """Interface principale pour le 3D GAN"""
    
    def __init__(self, config_path='configs/config.json', checkpoint_path=None, device=None):
        """
        Initialize the 3D GAN interface
        
        Args:
            config_path: path to configuration file
            checkpoint_path: path to pre-trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        self.config = load_config(config_path)
        print(f"Configuration loaded from {config_path}")
        
        # Initialize models
        self.init_models()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print("No checkpoint provided - model using random weights")
        
        # Create output directory
        self.output_dir = self.config['export']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
    
    def init_models(self):
        """Initialize generator and optional image encoder"""
        # Generator
        self.generator = Generator(
            latent_dim=self.config['model']['latent_dim'],
            num_points=self.config['model']['num_points'],
            layers=self.config['model']['generator']['layers']
        ).to(self.device)
        self.generator.eval()
        
        # Optional: Image encoder for image-to-3D
        if self.config['image_to_3d']['enabled']:
            self.image_encoder = models.resnet50(
                pretrained=self.config['image_to_3d']['pretrained']
            )
            # Remove classification layer
            self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
            self.image_encoder = self.image_encoder.to(self.device)
            self.image_encoder.eval()
            
            # Conditional generator
            self.conditional_generator = ConditionalGenerator(
                latent_dim=self.config['model']['latent_dim'],
                num_points=self.config['model']['num_points'],
                image_feature_dim=2048
            ).to(self.device)
            self.conditional_generator.eval()
            
            print("Image encoder initialized for image-to-3D generation")
    
    def load_checkpoint(self, checkpoint_path):
        """Load pre-trained model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    
    def generate_random(self, num_samples=1, latent_dim=None):
        """
        Generate random 3D models
        
        Args:
            num_samples: number of models to generate
            latent_dim: dimension of latent vector (uses config if None)
            
        Returns:
            point_clouds: tensor of shape (num_samples, num_points, 3)
        """
        if latent_dim is None:
            latent_dim = self.config['model']['latent_dim']
        
        with torch.no_grad():
            z = torch.randn(num_samples, latent_dim).to(self.device)
            point_clouds = self.generator(z)
        
        return point_clouds
    
    def image_to_3d(self, image_path, num_samples=1):
        """
        Generate 3D model from image
        
        Args:
            image_path: path to input image
            num_samples: number of variations to generate
            
        Returns:
            point_clouds: tensor of shape (num_samples, num_points, 3)
        """
        if not self.config['image_to_3d']['enabled']:
            raise RuntimeError("Image-to-3D is not enabled. Set 'enabled': true in config")
        
        if not hasattr(self, 'image_encoder'):
            self.init_models()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract image features
        with torch.no_grad():
            image_features = self.image_encoder(image_tensor)
            image_features = image_features.view(image_features.size(0), -1)
            
            # Generate 3D models with variations
            point_clouds = []
            for _ in range(num_samples):
                z = torch.randn(1, self.config['model']['latent_dim']).to(self.device)
                pc = self.conditional_generator(z, image_features)
                point_clouds.append(pc)
            
            point_clouds = torch.cat(point_clouds, dim=0)
        
        return point_clouds
    
    def export_obj(self, point_clouds, names=None, scale=1.0):
        """
        Export point clouds as OBJ files
        
        Args:
            point_clouds: tensor of shape (num_samples, num_points, 3) or list
            names: list of filenames (optional)
            scale: scaling factor
            
        Returns:
            paths: list of saved file paths
        """
        paths = []
        
        if isinstance(point_clouds, torch.Tensor):
            point_clouds = [pc for pc in point_clouds]
        
        for i, pc in enumerate(point_clouds):
            # Generate filename
            if names and i < len(names):
                filename = names[i]
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_{timestamp}_{i:03d}.obj"
            
            filepath = os.path.join(self.output_dir, filename)
            PointCloudToOBJ.save_point_cloud_as_obj(pc, filepath, scale=scale)
            paths.append(filepath)
            print(f"Saved: {filepath}")
        
        return paths
    
    def reconstruct_mesh(self, point_clouds, method='poisson'):
        """
        Reconstruct mesh from point clouds
        
        Args:
            point_clouds: tensor or list of point clouds
            method: 'poisson' or 'ball_pivoting'
            
        Returns:
            meshes: list of mesh objects
        """
        meshes = []
        
        if isinstance(point_clouds, torch.Tensor):
            point_clouds = [pc for pc in point_clouds]
        
        for pc in point_clouds:
            try:
                if method == 'poisson':
                    mesh = PointCloudToMesh.poisson_reconstruction(pc)
                elif method == 'ball_pivoting':
                    mesh = PointCloudToMesh.ball_pivoting(pc)
                else:
                    raise ValueError(f"Unknown method: {method}")
                meshes.append(mesh)
            except Exception as e:
                print(f"Mesh reconstruction failed: {e}")
        
        return meshes
    
    def export_fbx(self, point_clouds, names=None, method='poisson'):
        """
        Export point clouds as FBX files (with mesh reconstruction)
        
        Args:
            point_clouds: tensor of shape (num_samples, num_points, 3)
            names: list of filenames (optional)
            method: 'poisson' or 'ball_pivoting'
            
        Returns:
            paths: list of saved file paths
        """
        paths = []
        
        # First reconstruct meshes
        meshes = self.reconstruct_mesh(point_clouds, method=method)
        
        for i, mesh in enumerate(meshes):
            # Generate filename
            if names and i < len(names):
                filename = names[i].replace('.obj', '.fbx')
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_{timestamp}_{i:03d}.fbx"
            
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                OBJExporter.save_mesh_as_obj(mesh, filepath.replace('.fbx', '.obj'))
                FBXExporter.obj_to_fbx(filepath.replace('.fbx', '.obj'), filepath)
                paths.append(filepath)
                print(f"Saved: {filepath}")
            except Exception as e:
                print(f"FBX export failed: {e}")
        
        return paths
    
    def generate_and_export(self, num_models=5, export_format='obj', 
                           mesh_reconstruction=False, output_names=None):
        """
        Generate models and export directly
        
        Args:
            num_models: number of models to generate
            export_format: 'obj' or 'fbx'
            mesh_reconstruction: whether to reconstruct mesh
            output_names: custom output names
            
        Returns:
            paths: list of saved file paths
        """
        print(f"\nGenerating {num_models} random 3D models...")
        point_clouds = self.generate_random(num_samples=num_models)
        
        if export_format == 'obj':
            print("Exporting as OBJ files...")
            paths = self.export_obj(point_clouds, names=output_names)
        elif export_format == 'fbx':
            print("Exporting as FBX files...")
            paths = self.export_fbx(point_clouds, names=output_names)
        else:
            raise ValueError(f"Unknown format: {export_format}")
        
        return paths
    
    def image_to_3d_export(self, image_path, num_variations=1, export_format='obj'):
        """
        Generate 3D model from image and export
        
        Args:
            image_path: path to input image
            num_variations: number of variations to generate
            export_format: 'obj' or 'fbx'
            
        Returns:
            paths: list of saved file paths
        """
        print(f"\nGenerating 3D model from image: {image_path}")
        point_clouds = self.image_to_3d(image_path, num_samples=num_variations)
        
        # Use image filename as base
        base_name = Path(image_path).stem
        names = [f"{base_name}_variation_{i}.{export_format}" for i in range(num_variations)]
        
        if export_format == 'obj':
            print("Exporting as OBJ files...")
            paths = self.export_obj(point_clouds, names=names)
        elif export_format == 'fbx':
            print("Exporting as FBX files...")
            paths = self.export_fbx(point_clouds, names=names)
        else:
            raise ValueError(f"Unknown format: {export_format}")
        
        return paths


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description='3D GAN - Generate 3D Models with AI'
    )
    
    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'image', 'batch'],
                        help='Generation mode: random, image-to-3D, or batch')
    
    parser.add_argument('--num-models', type=int, default=5,
                        help='Number of models to generate (for random mode)')
    
    parser.add_argument('--image', type=str,
                        help='Path to input image (for image-to-3D mode)')
    
    parser.add_argument('--num-variations', type=int, default=3,
                        help='Number of variations from image (for image mode)')
    
    parser.add_argument('--format', type=str, default='obj',
                        choices=['obj', 'fbx'],
                        help='Export format')
    
    parser.add_argument('--output-dir', type=str, default='./generated_models',
                        help='Output directory for generated models')
    
    parser.add_argument('--checkpoint', type=str,
                        help='Path to pre-trained model checkpoint')
    
    parser.add_argument('--config', type=str, default='configs/config.json',
                        help='Path to configuration file')
    
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = ThreeDGANInterface(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Update output directory
    interface.output_dir = args.output_dir
    os.makedirs(interface.output_dir, exist_ok=True)
    
    # Execute mode
    if args.mode == 'random':
        print(f"\n{'='*60}")
        print("Mode: Random 3D Model Generation")
        print(f"{'='*60}")
        interface.generate_and_export(
            num_models=args.num_models,
            export_format=args.format
        )
    
    elif args.mode == 'image':
        if not args.image:
            print("Error: --image is required for image mode")
            return
        
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return
        
        print(f"\n{'='*60}")
        print("Mode: Image-to-3D Generation")
        print(f"{'='*60}")
        interface.image_to_3d_export(
            image_path=args.image,
            num_variations=args.num_variations,
            export_format=args.format
        )
    
    elif args.mode == 'batch':
        print(f"\n{'='*60}")
        print("Mode: Batch Generation")
        print(f"{'='*60}")
        # Generate both random and image-based models
        interface.generate_and_export(
            num_models=args.num_models,
            export_format=args.format
        )
    
    print(f"\n{'='*60}")
    print(f"Generation complete! Models saved to: {interface.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
