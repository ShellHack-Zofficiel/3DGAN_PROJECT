#!/usr/bin/env python3
"""Script de test: lance un court entraînement (smoke test) pour vérifier la pipeline."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from training.train import Trainer
import numpy as np

def create_synthetic_loader(batch_size, num_points, num_samples=64):
    # Create random point clouds in [-1,1]
    data = np.random.uniform(-1, 1, size=(num_samples, num_points, 3)).astype(np.float32)
    tensor = torch.from_numpy(data)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def main():
    trainer = Trainer(config_path='configs/config.json', device='cpu')

    cfg = trainer.config
    batch_size = cfg['data'].get('batch_size', 8)
    num_points = cfg['model'].get('num_points', 2048)

    train_loader = create_synthetic_loader(batch_size, num_points, num_samples=32)
    val_loader = create_synthetic_loader(batch_size, num_points, num_samples=8)

    # Run a very short training (2 epochs) as smoke test
    trainer.train(train_loader, val_loader=val_loader, num_epochs=2)

if __name__ == '__main__':
    main()
