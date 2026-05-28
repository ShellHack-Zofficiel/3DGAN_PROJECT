import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from training.train import Trainer
from training.utils import PointCloudDataset, load_config


def main():
    # Projet root: si DRIVE_PROJECT_PATH est défini, on l'utilise
    project_root = Path(os.environ.get('DRIVE_PROJECT_PATH', Path.cwd()))
    print(f"Using project root: {project_root}")

    config_path = project_root / 'configs' / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    data_dir = project_root / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading dataset from: {data_dir}")
    dataset = PointCloudDataset(str(data_dir), num_points=config['model']['num_points'])
    dataset_length = len(dataset)
    print(f"Dataset size: {dataset_length} samples")

    if dataset_length == 0:
        raise ValueError(
            f"No point cloud files found in {data_dir}. Add .npy or .pt files to the data directory "
            "or generate synthetic data before training."
        )

    val_split = config['data'].get('val_split', 0.1)
    val_size = max(1, int(dataset_length * val_split))
    train_size = dataset_length - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=torch.cuda.is_available()
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    trainer = Trainer(config_path=str(config_path), device=device)
    trainer.train(train_loader, val_loader=val_loader, num_epochs=config['training']['epochs'])


if __name__ == '__main__':
    main()
