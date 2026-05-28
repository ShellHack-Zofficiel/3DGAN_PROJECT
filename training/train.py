import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.generator import Generator, ConditionalGenerator
from models.discriminator import Discriminator, PointNetDiscriminator
from models.losses import ChamferDistance, GANLoss, CombinedLoss
from training.utils import (
    PointCloudDataset, 
    EarlyStopping, 
    AverageMeter,
    load_config,
    save_checkpoint,
    load_checkpoint,
    set_seed
)
from export.to_obj import PointCloudToOBJ


class Trainer:
    """Classe d'entraînement pour le 3D GAN"""
    
    def __init__(self, config_path='configs/config.json', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Set random seed
        set_seed(42)
        
        # Initialize models
        self.init_models()
        
        # Initialize optimizers
        self.init_optimizers()
        
        # Initialize losses
        self.init_losses()
        
        # Create checkpoint directory
        os.makedirs(self.config['training']['checkpoint_dir'], exist_ok=True)
        
        print("Trainer initialized successfully!")
    
    def init_models(self):
        """Initialize generator and discriminator"""
        # Generator
        self.generator = Generator(
            latent_dim=self.config['model']['latent_dim'],
            num_points=self.config['model']['num_points'],
            layers=self.config['model']['generator']['layers']
        ).to(self.device)
        
        # Discriminator
        self.discriminator = PointNetDiscriminator(
            num_points=self.config['model']['num_points'],
            use_sigmoid=not self.config['training'].get('use_wasserstein', False),
            use_spectral_norm=self.config['model']['discriminator'].get('use_spectral_norm', True)
        ).to(self.device)
        
        print(f"Generator: {self.generator}")
        print(f"Discriminator: {self.discriminator}")
    
    def init_optimizers(self):
        """Initialize optimizers"""
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config['training']['learning_rate_g'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['training']['learning_rate_d'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2'])
        )
        # Learning rate schedulers (ReduceLROnPlateau tuned for GAN training)
        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_g, mode='min', factor=0.5, patience=20
        )
        self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_d, mode='min', factor=0.5, patience=20
        )
    
    def init_losses(self):
        """Initialiser les fonctions de perte"""
        self.use_wasserstein = self.config['training'].get('use_wasserstein', False)
        self.gradient_penalty = self.config['training'].get('gradient_penalty', False)
        self.lambda_gp = self.config['training'].get('lambda_gp', 10.0)
        gan_weight = self.config['training'].get('gan_weight', 1.0)
        chamfer_weight = self.config['training'].get('chamfer_weight', 1.0)

        self.gan_loss = GANLoss()
        self.chamfer_loss = ChamferDistance()
        self.combined_loss = CombinedLoss(
            gan_weight=gan_weight,
            chamfer_weight=chamfer_weight,
            use_wasserstein=self.use_wasserstein
        )

        # Sample directory for validation samples during training
        self.sample_dir = os.path.join(self.config['training'].get('log_dir', './logs'), 'samples')
        os.makedirs(self.sample_dir, exist_ok=True)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        g_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        
        for batch_idx, real_pc in enumerate(train_loader):
            # Support DataLoader returning tuples (TensorDataset) or raw tensors
            if isinstance(real_pc, (list, tuple)):
                real_pc = real_pc[0]
            real_pc = real_pc.to(self.device)
            batch_size = real_pc.size(0)
            
            # ==================== Train Discriminator ====================
            self.optimizer_d.zero_grad()
            
            # Real samples
            real_output = self.discriminator(real_pc)
            
            # Fake samples
            z = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
            fake_pc = self.generator(z)
            fake_output = self.discriminator(fake_pc.detach())
            
            # Perte du discriminateur: distinguer réel/faux
            d_loss = self.combined_loss.discriminator_loss(fake_output, real_output)
            if self.use_wasserstein and self.gradient_penalty:
                gp = self.combined_loss.gradient_penalty(self.discriminator, real_pc, fake_pc.detach(), self.device)
                d_loss = d_loss + self.lambda_gp * gp
            d_loss.backward()
            self.optimizer_d.step()
            
            d_loss_meter.update(d_loss.item(), batch_size)
            
            # ==================== Train Generator ====================
            self.optimizer_g.zero_grad()
            
            # Generate fake samples
            z = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
            fake_pc = self.generator(z)
            fake_output = self.discriminator(fake_pc)
            
            # Perte du générateur: adversariale + Chamfer
            total_g_loss = self.combined_loss.generator_loss(real_pc, fake_pc, fake_output)
            
            total_g_loss.backward()
            self.optimizer_g.step()
            
            g_loss_meter.update(total_g_loss.item(), batch_size)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"D_loss: {d_loss_meter.avg:.4f}, "
                      f"G_loss: {g_loss_meter.avg:.4f}")
        
        return g_loss_meter.avg, d_loss_meter.avg
    
    def validate(self, val_loader):
        """Validate the model"""
        self.generator.eval()
        self.discriminator.eval()
        
        g_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for real_pc in val_loader:
                if isinstance(real_pc, (list, tuple)):
                    real_pc = real_pc[0]
                real_pc = real_pc.to(self.device)
                batch_size = real_pc.size(0)
                
                # Generate fake samples
                z = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
                fake_pc = self.generator(z)
                
                # Discriminator output
                real_output = self.discriminator(real_pc)
                fake_output = self.discriminator(fake_pc)
                
                # Pertes de validation
                d_loss = self.combined_loss.discriminator_loss(fake_output, real_output)
                total_g_loss = self.combined_loss.generator_loss(real_pc, fake_pc, fake_output)
                
                d_loss_meter.update(d_loss.item(), batch_size)
                g_loss_meter.update(total_g_loss.item(), batch_size)
        
        return g_loss_meter.avg, d_loss_meter.avg

    def save_sample_point_clouds(self, epoch, num_samples=4):
        """Générer et sauvegarder des échantillons pour suivi visuel"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['model']['latent_dim']).to(self.device)
            samples = self.generator(z)

        paths = []
        for i, pc in enumerate(samples):
            filename = f"sample_epoch_{epoch:04d}_{i:02d}.obj"
            out_path = os.path.join(self.sample_dir, filename)
            PointCloudToOBJ.save_point_cloud_as_obj(pc, out_path)
            paths.append(out_path)

        print(f"Saved validation samples: {paths}")
    
    def train(self, train_loader, val_loader=None, num_epochs=None):
        """Train the model"""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        early_stopping = EarlyStopping(patience=20, min_delta=0.0001)
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"{'='*50}")
            
            # Train
            g_loss, d_loss = self.train_epoch(train_loader)
            print(f"Training - D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
            
            # Validate
            if val_loader:
                val_g_loss, val_d_loss = self.validate(val_loader)
                print(f"Validation - D_loss: {val_d_loss:.4f}, G_loss: {val_g_loss:.4f}")
                
                # Early stopping
                early_stopping(val_g_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break
                # Save sample point clouds for inspection
                self.save_sample_point_clouds(epoch+1, num_samples=4)
                # Step schedulers based on validation generator loss
                self.scheduler_g.step(val_g_loss)
                self.scheduler_d.step(val_d_loss)
            else:
                # No validation: step schedulers on training generator loss
                self.scheduler_g.step(g_loss)
                self.scheduler_d.step(d_loss)
            
            # Enregistrer un checkpoint complet tous les save_interval epochs
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['training']['checkpoint_dir'],
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                save_checkpoint(
                    self.generator,
                    self.optimizer_g,
                    epoch,
                    g_loss,
                    checkpoint_path,
                    discriminator=self.discriminator,
                    optimizer_d=self.optimizer_d
                )
                print(f"Checkpoint saved: {checkpoint_path}")
    
    def generate_samples(self, num_samples=10):
        """Generate samples"""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['model']['latent_dim']).to(self.device)
            samples = self.generator(z)
        
        return samples


if __name__ == "__main__":
    # Initialize trainer
    trainer = Trainer(config_path='configs/config.json')
    
    # Create dummy data loader for testing
    print("\nTraining script ready!")
    print("To train the model:")
    print("1. Prepare your dataset in the 'data/' directory")
    print("2. Run: python training/train.py")
