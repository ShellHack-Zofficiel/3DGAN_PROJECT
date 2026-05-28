"""
3DGAN Project - Generative Adversarial Network for 3D Model Generation

A complete framework for generating realistic 3D models using GANs:
- Generate random 3D point clouds
- Image-to-3D generation
- Mesh reconstruction (Poisson, Ball Pivoting)
- Export to OBJ and FBX formats

Example usage:
    from main import ThreeDGANInterface
    
    gan = ThreeDGANInterface()
    
    # Generate random models
    models = gan.generate_random(num_samples=5)
    gan.export_obj(models)
    
    # Generate from image
    models = gan.image_to_3d('image.jpg', num_samples=3)
    gan.export_fbx(models)

Author: 3DGAN_Project
License: MIT
"""

__version__ = "1.0.0"
__author__ = "3DGAN_Project"
__license__ = "MIT"

# Import main components
from main import ThreeDGANInterface
from models import Generator, ConditionalGenerator, Discriminator, PointNetDiscriminator
from training import Trainer

__all__ = [
    'ThreeDGANInterface',
    'Generator',
    'ConditionalGenerator',
    'Discriminator',
    'PointNetDiscriminator',
    'Trainer',
]
