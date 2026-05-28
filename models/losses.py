import torch
import torch.nn as nn
import numpy as np


class ChamferDistance(nn.Module):
    """Perte de distance de Chamfer pour comparer deux nuages de points"""
    
    def __init__(self):
        super(ChamferDistance, self).__init__()
    
    def forward(self, pc1, pc2):
        """
        Calculate Chamfer distance between two point clouds
        
        Args:
            pc1: point cloud 1 of shape (batch_size, num_points1, 3)
            pc2: point cloud 2 of shape (batch_size, num_points2, 3)
            
        Returns:
            distance: scalar tensor representing the Chamfer distance
        """
        # Calculate pairwise distances
        # pc1: (B, N1, 3), pc2: (B, N2, 3)
        
        # Expand dimensions for broadcasting
        # pc1_expanded: (B, N1, 1, 3)
        # pc2_expanded: (B, 1, N2, 3)
        pc1_expanded = pc1.unsqueeze(2)  # (B, N1, 1, 3)
        pc2_expanded = pc2.unsqueeze(1)  # (B, 1, N2, 3)
        
        # Calculate squared distances
        diff = pc1_expanded - pc2_expanded  # (B, N1, N2, 3)
        dist_squared = torch.sum(diff ** 2, dim=-1)  # (B, N1, N2)
        
        # Minimum distances from pc1 to pc2
        dist_pc1_to_pc2 = torch.min(dist_squared, dim=2)[0]  # (B, N1)
        
        # Minimum distances from pc2 to pc1
        dist_pc2_to_pc1 = torch.min(dist_squared, dim=1)[0]  # (B, N2)
        
        # Chamfer distance
        chamfer_dist = torch.mean(dist_pc1_to_pc2) + torch.mean(dist_pc2_to_pc1)
        
        return chamfer_dist


class EMDLoss(nn.Module):
    """Earth Mover's Distance (Wasserstein distance) for point clouds"""
    
    def __init__(self):
        super(EMDLoss, self).__init__()
    
    def forward(self, pc1, pc2):
        """
        Simplified EMD loss (Hungarian algorithm would be needed for true EMD)
        Using approximate version based on Chamfer distance
        
        Args:
            pc1: point cloud 1
            pc2: point cloud 2
            
        Returns:
            loss: scalar tensor
        """
        # Simplified version - use Chamfer distance
        return ChamferDistance()(pc1, pc2)


class GANLoss(nn.Module):
    """Standard GAN loss (Binary Cross Entropy)"""
    
    def __init__(self):
        super(GANLoss, self).__init__()
        self.bce = nn.BCELoss()
    
    def forward_generator(self, discriminator_output):
        """
        Generator loss: fool the discriminator
        
        Args:
            discriminator_output: discriminator output for fake samples
            
        Returns:
            loss: scalar tensor
        """
        # Generator wants discriminator to output 1 (think it's real)
        target = torch.ones_like(discriminator_output)
        return self.bce(discriminator_output, target)
    
    def forward_discriminator(self, fake_output, real_output):
        """
        Discriminator loss: distinguish real from fake
        
        Args:
            fake_output: discriminator output for fake samples
            real_output: discriminator output for real samples
            
        Returns:
            loss: scalar tensor
        """
        fake_target = torch.zeros_like(fake_output)
        real_target = torch.ones_like(real_output)
        
        fake_loss = self.bce(fake_output, fake_target)
        real_loss = self.bce(real_output, real_target)
        
        return fake_loss + real_loss


class WassersteinLoss(nn.Module):
    """Wasserstein GAN loss"""
    
    def forward_generator(self, discriminator_output):
        """Generator loss"""
        return -torch.mean(discriminator_output)
    
    def forward_discriminator(self, fake_output, real_output):
        """Discriminator loss"""
        return torch.mean(fake_output) - torch.mean(real_output)


class CombinedLoss(nn.Module):
    """Combined loss: GAN loss + Chamfer distance"""
    
    def __init__(self, gan_weight=1.0, chamfer_weight=1.0, use_wasserstein=False):
        super(CombinedLoss, self).__init__()
        self.gan_weight = gan_weight
        self.chamfer_weight = chamfer_weight
        self.use_wasserstein = use_wasserstein
        
        if use_wasserstein:
            self.gan_loss = WassersteinLoss()
        else:
            self.gan_loss = GANLoss()
        
        self.chamfer_loss = ChamferDistance()

    def gradient_penalty(self, discriminator, real_pc, fake_pc, device='cpu'):
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_pc.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=device)
        alpha = alpha.expand_as(real_pc)
        interpolates = alpha * real_pc + (1 - alpha) * fake_pc
        interpolates.requires_grad_(True)

        disc_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty
    
    def generator_loss(self, real_pc, fake_pc, discriminator_output):
        """
        Total generator loss
        
        Args:
            real_pc: real point cloud
            fake_pc: generated point cloud
            discriminator_output: discriminator output for fake samples
            
        Returns:
            loss: scalar tensor
        """
        gan_loss = self.gan_loss.forward_generator(discriminator_output)
        chamfer = self.chamfer_loss(fake_pc, real_pc)
        
        return self.gan_weight * gan_loss + self.chamfer_weight * chamfer
    
    def discriminator_loss(self, fake_output, real_output):
        """Discriminator loss"""
        return self.gan_loss.forward_discriminator(fake_output, real_output)
