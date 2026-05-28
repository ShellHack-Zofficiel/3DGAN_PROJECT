import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path


class PointCloudDataset(Dataset):
    """Dataset pour charger des nuages de points"""
    
    def __init__(self, data_dir, num_points=2048, transform=None):
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        
        # Find all point cloud files
        self.files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npy') or file.endswith('.pt'):
                    self.files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load point cloud
        if file_path.endswith('.npy'):
            points = np.load(file_path)
        else:  # .pt
            points = torch.load(file_path).numpy()
        
        # Ensure correct number of points
        if len(points) > self.num_points:
            # Randomly sample points
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            # Pad with repetition
            num_to_add = self.num_points - len(points)
            indices = np.random.choice(len(points), num_to_add, replace=True)
            points = np.vstack([points, points[indices]])
        
        # Convert to tensor
        points = torch.from_numpy(points).float()
        
        # Apply transform if any
        if self.transform:
            points = self.transform(points)
        
        return points


class PointCloudPreprocessor:
    """Classe utilitaire pour prétraiter des nuages de points"""
    
    @staticmethod
    def normalize_point_cloud(points, center=True, scale=True):
        """
        Normalize point cloud to [-1, 1]
        
        Args:
            points: array of shape (N, 3)
            center: whether to center the cloud
            scale: whether to scale the cloud
            
        Returns:
            normalized_points: normalized array
        """
        points = np.copy(points)
        
        if center:
            centroid = np.mean(points, axis=0)
            points -= centroid
        
        if scale:
            # Scale to fit in [-1, 1]
            max_distance = np.max(np.abs(points))
            if max_distance > 0:
                points /= max_distance
        
        return points
    
    @staticmethod
    def normalize_batch(batch, center=True, scale=True):
        """Normalize a batch of point clouds"""
        return torch.stack([
            torch.from_numpy(PointCloudPreprocessor.normalize_point_cloud(
                pc.numpy() if isinstance(pc, torch.Tensor) else pc,
                center=center, scale=scale
            )).float()
            for pc in batch
        ])
    
    @staticmethod
    def augment_point_cloud(points, rotation=True, scaling=True, jitter=True):
        """
        Augment point cloud with random transformations
        
        Args:
            points: array of shape (N, 3)
            rotation: whether to apply random rotation
            scaling: whether to apply random scaling
            jitter: whether to add noise
            
        Returns:
            augmented_points: augmented array
        """
        points = np.copy(points)
        
        if rotation:
            # Random rotation around z-axis
            angle = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            points = points @ rotation_matrix.T
        
        if scaling:
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            points *= scale
        
        if jitter:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.01, points.shape)
            points += noise
        
        return points


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class AverageMeter:
    """Compute running average"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_checkpoint(model, optimizer, epoch, loss, save_path, discriminator=None, optimizer_d=None):
    """Enregistrer un checkpoint complet"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if discriminator is not None:
        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
    if optimizer_d is not None:
        checkpoint['optimizer_d_state_dict'] = optimizer_d.state_dict()
    
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, checkpoint_path, device='cuda', discriminator=None, optimizer_d=None):
    """Charger un checkpoint complet"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if discriminator is not None and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    if optimizer_d is not None and 'optimizer_d_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    
    return model, optimizer, epoch, loss


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
