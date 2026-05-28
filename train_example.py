#!/usr/bin/env python3
"""
Complete Training Workflow Example
Exemple d'entraînement complet du modèle 3D GAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import json
from pathlib import Path

# Import des modules du projet
from models.generator import Generator
from models.discriminator import PointNetDiscriminator
from models.losses import GANLoss, ChamferDistance
from training.utils import (
    PointCloudDataset,
    PointCloudPreprocessor,
    EarlyStopping,
    AverageMeter,
    load_config,
    save_checkpoint,
    set_seed
)


class TrainingWorkflow:
    """Workflow complet d'entraînement"""
    
    def __init__(self, config_path='configs/config.json', device=None):
        """
        Initialiser le workflow d'entraînement
        
        Args:
            config_path: chemin vers le fichier de configuration
            device: 'cuda' ou 'cpu'
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n{'='*70}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        # Charger la configuration
        self.config = load_config(config_path)
        print(f"Configuration chargée depuis: {config_path}")
        
        # Définir la graine aléatoire
        set_seed(42)
        
        # Créer les répertoires nécessaires
        self._create_directories()
        
        # Initialiser les modèles
        self._init_models()
        
        # Initialiser les optimiseurs
        self._init_optimizers()
        
        # Initialiser les pertes
        self._init_losses()
    
    def _create_directories(self):
        """Créer les répertoires nécessaires"""
        dirs = [
            self.config['training']['checkpoint_dir'],
            self.config['training']['log_dir'],
            self.config['export']['output_dir'],
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ {dir_path}")
    
    def _init_models(self):
        """Initialiser le générateur et discriminateur"""
        self.generator = Generator(
            latent_dim=self.config['model']['latent_dim'],
            num_points=self.config['model']['num_points'],
            layers=self.config['model']['generator']['layers']
        ).to(self.device)
        
        self.discriminator = PointNetDiscriminator(
            num_points=self.config['model']['num_points']
        ).to(self.device)
        
        print(f"\n✓ Générateur créé")
        print(f"  - Latent dim: {self.config['model']['latent_dim']}")
        print(f"  - Num points: {self.config['model']['num_points']}")
        
        print(f"\n✓ Discriminateur créé")
    
    def _init_optimizers(self):
        """Initialiser les optimiseurs"""
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
        
        print(f"\n✓ Optimiseurs créés")
        print(f"  - G LR: {self.config['training']['learning_rate_g']}")
        print(f"  - D LR: {self.config['training']['learning_rate_d']}")
    
    def _init_losses(self):
        """Initialiser les fonctions de perte"""
        self.gan_loss = GANLoss()
        self.chamfer_loss = ChamferDistance()
        
        print(f"\n✓ Fonctions de perte créées")
    
    def prepare_dataset(self, data_dir='data/', train_ratio=0.8, val_ratio=0.1):
        """
        Préparer les données d'entraînement
        
        Args:
            data_dir: répertoire contenant les données
            train_ratio: proportion d'entraînement
            val_ratio: proportion de validation
            
        Returns:
            train_loader, val_loader, test_loader: DataLoaders
        """
        print(f"\n{'='*70}")
        print("Préparation des données")
        print(f"{'='*70}\n")
        
        # Créer le dataset
        dataset = PointCloudDataset(
            data_dir=data_dir,
            num_points=self.config['model']['num_points']
        )
        
        print(f"✓ Dataset créé avec {len(dataset)} échantillons")
        
        # Calculer les tailles
        train_size = int(len(dataset) * train_ratio)
        val_size = int(len(dataset) * val_ratio)
        test_size = len(dataset) - train_size - val_size
        
        # Diviser le dataset
        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size]
        )
        
        print(f"  - Train: {train_size}")
        print(f"  - Val: {val_size}")
        print(f"  - Test: {test_size}")
        
        # Créer les DataLoaders
        train_loader = DataLoader(
            train_set,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        val_loader = DataLoader(
            val_set,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        test_loader = DataLoader(
            test_set,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        print(f"\n✓ DataLoaders créés")
        print(f"  - Batch size: {self.config['data']['batch_size']}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader, epoch):
        """
        Entraîner une époque
        
        Args:
            train_loader: DataLoader d'entraînement
            epoch: numéro de l'époque
            
        Returns:
            g_loss_avg, d_loss_avg: pertes moyennes
        """
        self.generator.train()
        self.discriminator.train()
        
        g_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        
        for batch_idx, real_pc in enumerate(train_loader):
            real_pc = real_pc.to(self.device)
            batch_size = real_pc.size(0)
            
            # ==================== Entraîner Discriminateur ====================
            self.optimizer_d.zero_grad()
            
            # Samples réels
            real_output = self.discriminator(real_pc)
            
            # Samples faux
            z = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
            fake_pc = self.generator(z)
            fake_output = self.discriminator(fake_pc.detach())
            
            # Perte du discriminateur
            d_loss = self.gan_loss.forward_discriminator(fake_output, real_output)
            d_loss.backward()
            self.optimizer_d.step()
            
            d_loss_meter.update(d_loss.item(), batch_size)
            
            # ==================== Entraîner Générateur ====================
            self.optimizer_g.zero_grad()
            
            # Générer samples faux
            z = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
            fake_pc = self.generator(z)
            fake_output = self.discriminator(fake_pc)
            
            # Perte du générateur
            gan_loss = self.gan_loss.forward_generator(fake_output)
            chamfer = self.chamfer_loss(fake_pc, real_pc)
            total_g_loss = gan_loss + chamfer
            
            total_g_loss.backward()
            self.optimizer_g.step()
            
            g_loss_meter.update(total_g_loss.item(), batch_size)
            
            # Afficher la progression
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"D_loss: {d_loss_meter.avg:.4f} G_loss: {g_loss_meter.avg:.4f}")
        
        return g_loss_meter.avg, d_loss_meter.avg
    
    def validate(self, val_loader):
        """
        Valider le modèle
        
        Args:
            val_loader: DataLoader de validation
            
        Returns:
            g_loss_avg, d_loss_avg: pertes moyennes
        """
        self.generator.eval()
        self.discriminator.eval()
        
        g_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        
        with torch.no_grad():
            for real_pc in val_loader:
                real_pc = real_pc.to(self.device)
                batch_size = real_pc.size(0)
                
                # Générer samples
                z = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
                fake_pc = self.generator(z)
                
                # Discriminateur outputs
                real_output = self.discriminator(real_pc)
                fake_output = self.discriminator(fake_pc)
                
                # Pertes
                d_loss = self.gan_loss.forward_discriminator(fake_output, real_output)
                gan_loss = self.gan_loss.forward_generator(fake_output)
                chamfer = self.chamfer_loss(fake_pc, real_pc)
                total_g_loss = gan_loss + chamfer
                
                d_loss_meter.update(d_loss.item(), batch_size)
                g_loss_meter.update(total_g_loss.item(), batch_size)
        
        return g_loss_meter.avg, d_loss_meter.avg
    
    def train(self, train_loader, val_loader=None):
        """
        Entraîner le modèle complet
        
        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation (optionnel)
        """
        print(f"\n{'='*70}")
        print("Début de l'entraînement")
        print(f"{'='*70}\n")
        
        num_epochs = self.config['training']['epochs']
        early_stopping = EarlyStopping(patience=20, min_delta=0.0001)
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Époque [{epoch+1}/{num_epochs}]")
            print(f"{'='*70}")
            
            # Entraîner
            g_loss, d_loss = self.train_epoch(train_loader, epoch+1)
            print(f"\nEntraînement - D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}")
            
            # Valider
            if val_loader:
                val_g_loss, val_d_loss = self.validate(val_loader)
                print(f"Validation - D_loss: {val_d_loss:.4f} G_loss: {val_g_loss:.4f}")
                
                # Early stopping
                early_stopping(val_g_loss)
                if early_stopping.early_stop:
                    print("\nEarly stopping déclenché!")
                    break
            
            # Sauvegarder checkpoint
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
                    checkpoint_path
                )
                print(f"✓ Checkpoint sauvegardé: {checkpoint_path}")
        
        print(f"\n{'='*70}")
        print("Entraînement terminé!")
        print(f"{'='*70}\n")


def main():
    """Exemple d'utilisation du workflow d'entraînement"""
    
    # Créer le workflow
    workflow = TrainingWorkflow(config_path='configs/config.json')
    
    # Préparer les données
    print("\n⚠️  IMPORTANT: Assurez-vous que vos données sont dans data/")
    print("   Format: fichiers .npy ou .pt avec shapes (N, 3)\n")
    
    try:
        # Essayer de charger les données
        train_loader, val_loader, test_loader = workflow.prepare_dataset(data_dir='data/')
        
        # Entraîner le modèle
        workflow.train(train_loader, val_loader)
        
        print("\n✅ Entraînement réussi!")
        print(f"Modèles sauvegardés dans: {workflow.config['training']['checkpoint_dir']}")
        
    except FileNotFoundError:
        print("\n❌ Aucune donnée trouvée dans data/")
        print("\nPour utiliser ce script:")
        print("1. Télécharger ShapeNet ou ModelNet")
        print("2. Convertir les modèles en format .npy")
        print("3. Placer les fichiers dans data/")
        print("\nExemple de structure:")
        print("  data/")
        print("    ├── model_001.npy")
        print("    ├── model_002.npy")
        print("    └── ...\n")
        
        print("Pour tester sans données:")
        print("  python inference.py --mode random --num 5")


if __name__ == "__main__":
    main()
