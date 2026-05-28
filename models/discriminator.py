import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Réseau discriminateur pour valider les nuages de points"""
    
    def __init__(self, num_points=2048, layers=None, dropout=0.3):
        super(Discriminator, self).__init__()
        self.num_points = num_points
        
        if layers is None:
            layers = [128, 256, 512]
        
        # Input: (batch_size, num_points, 3) -> flatten to (batch_size, num_points*3)
        self.fc_layers = nn.ModuleList()
        prev_dim = num_points * 3
        
        for layer_dim in layers:
            self.fc_layers.append(nn.Linear(prev_dim, layer_dim))
            self.fc_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                self.fc_layers.append(nn.Dropout(dropout))
            prev_dim = layer_dim
        
        # Output layer: binary classification (real/fake)
        self.fc_output = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, point_cloud):
        """
        Forward pass
        
        Args:
            point_cloud: tensor of shape (batch_size, num_points, 3)
            
        Returns:
            validity: tensor of shape (batch_size, 1) with values in [0, 1]
        """
        batch_size = point_cloud.size(0)
        
        # Flatten point cloud
        x = point_cloud.view(batch_size, -1)
        
        # Pass through MLP layers
        for layer in self.fc_layers:
            x = layer(x)
        
        # Output validity score
        validity = self.sigmoid(self.fc_output(x))
        
        return validity


class PointNetDiscriminator(nn.Module):
    """Discriminator using PointNet architecture for better geometric understanding"""
    
    def __init__(self, num_points=2048, use_sigmoid=True, use_spectral_norm=True):
        super(PointNetDiscriminator, self).__init__()
        self.num_points = num_points
        self.use_sigmoid = use_sigmoid
        self.use_spectral_norm = use_spectral_norm
        
        # First MLP for per-point features
        self.fc1 = nn.utils.spectral_norm(nn.Linear(3, 64)) if use_spectral_norm else nn.Linear(3, 64)
        self.fc2 = nn.utils.spectral_norm(nn.Linear(64, 128)) if use_spectral_norm else nn.Linear(64, 128)
        self.fc3 = nn.utils.spectral_norm(nn.Linear(128, 256)) if use_spectral_norm else nn.Linear(128, 256)
        
        # Utilisation de LayerNorm pour les tenseurs 3D de points
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(256)
        
        # Global feature learning
        self.fc4 = nn.utils.spectral_norm(nn.Linear(512, 512)) if use_spectral_norm else nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Classification layers
        self.fc5 = nn.utils.spectral_norm(nn.Linear(512, 256)) if use_spectral_norm else nn.Linear(512, 256)
        self.fc6 = nn.utils.spectral_norm(nn.Linear(256, 128)) if use_spectral_norm else nn.Linear(256, 128)
        self.fc7 = nn.utils.spectral_norm(nn.Linear(128, 1)) if use_spectral_norm else nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, point_cloud):
        """
        Forward pass
        
        Args:
            point_cloud: tensor of shape (batch_size, num_points, 3)
            
        Returns:
            validity: tensor of shape (batch_size, 1) with values in [0, 1]
        """
        batch_size = point_cloud.size(0)
        
        # Extraction de caractéristiques par point
        x = self.relu(self.ln1(self.fc1(point_cloud)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.relu(self.ln3(self.fc3(x)))
        
        # Global pooling multi-échelle
        x_max = torch.max(x, dim=1)[0]
        x_avg = torch.mean(x, dim=1)
        x = torch.cat([x_max, x_avg], dim=1)  # (batch_size, 512)
        
        # Global feature learning
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        
        # Classification
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.relu(self.fc6(x))
        validity = self.fc7(x)
        
        if self.use_sigmoid:
            validity = self.sigmoid(validity)
        
        return validity
