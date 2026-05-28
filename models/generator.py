import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """Bloc MLP résiduel pour améliorer la stabilité et la profondeur."""

    def __init__(self, dim):
        super(ResidualMLPBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class Generator(nn.Module):
    """Réseau générateur pour la génération de nuages de points 3D."""
    
    def __init__(self, latent_dim=256, num_points=2048, layers=None):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        if layers is None:
            layers = [256, 512, 1024, 2048]
        
        self.fc_layers = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        prev_dim = latent_dim
        
        for layer_dim in layers:
            self.fc_layers.append(nn.Linear(prev_dim, layer_dim))
            self.fc_layers.append(nn.BatchNorm1d(layer_dim))
            self.fc_layers.append(nn.ReLU(inplace=True))
            self.res_blocks.append(ResidualMLPBlock(layer_dim))
            prev_dim = layer_dim
        
        self.fc_output = nn.Linear(prev_dim, num_points * 3)
        
    def forward(self, z):
        """
        Forward pass
        
        Args:
            z: latent vector of shape (batch_size, latent_dim)
            
        Returns:
            point_cloud: tensor of shape (batch_size, num_points, 3)
        """
        batch_size = z.size(0)
        
        x = z
        idx = 0
        for res_block in self.res_blocks:
            x = self.fc_layers[idx](x)
            x = self.fc_layers[idx + 1](x)
            x = self.fc_layers[idx + 2](x)
            x = res_block(x)
            idx += 3
        
        x = self.fc_output(x)
        
        # Reshape to (batch_size, num_points, 3)
        point_cloud = x.view(batch_size, self.num_points, 3)
        
        # Normalize to [-1, 1]
        point_cloud = torch.tanh(point_cloud)
        
        return point_cloud


class ConditionalGenerator(nn.Module):
    """Conditional Generator for image-to-3D generation"""
    
    def __init__(self, latent_dim=128, num_points=2048, image_feature_dim=2048):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        
        # Combine latent vector and image features
        combined_dim = latent_dim + image_feature_dim
        
        # MLP layers
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        
        # Output layer
        self.fc_output = nn.Linear(4096, num_points * 3)
    
    def forward(self, z, image_features):
        """
        Forward pass
        
        Args:
            z: latent vector of shape (batch_size, latent_dim)
            image_features: image features from encoder of shape (batch_size, image_feature_dim)
            
        Returns:
            point_cloud: tensor of shape (batch_size, num_points, 3)
        """
        batch_size = z.size(0)
        
        # Concatenate latent vector and image features
        x = torch.cat([z, image_features], dim=1)
        
        # Pass through MLP layers
        x = self.fc_layers(x)
        
        # Generate points
        x = self.fc_output(x)
        
        # Reshape to (batch_size, num_points, 3)
        point_cloud = x.view(batch_size, self.num_points, 3)
        
        # Normalize to [-1, 1]
        point_cloud = torch.tanh(point_cloud)
        
        return point_cloud
