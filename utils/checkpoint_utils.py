# checkpoint_utils.py

import os
import shutil
import torch

from datetime import datetime
from utils.validation import validate_model

def save_checkpoint(model, optimizer, scheduler, epoch, batch_step, config):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'batch_step': batch_step,
        'config': config
    }

    checkpoint_path = config['checkpoint_path']
    
    if os.path.exists(checkpoint_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(os.path.dirname(checkpoint_path), f"backup_{timestamp}")
        os.makedirs(backup_dir)

        backup_checkpoint_path = os.path.join(backup_dir, os.path.basename(checkpoint_path))
        shutil.move(checkpoint_path, backup_checkpoint_path)

        backup_dirs = sorted(
            [d for d in os.listdir(os.path.dirname(checkpoint_path)) if d.startswith("backup_")],
            key=lambda x: os.path.getmtime(os.path.join(os.path.dirname(checkpoint_path), x)),
            reverse=True
        )
        for old_backup in backup_dirs[5:]:
            shutil.rmtree(os.path.join(os.path.dirname(checkpoint_path), old_backup))

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    batch_step = checkpoint['batch_step']
    
    return epoch, batch_step, model, optimizer, scheduler

def save_checkpoint_and_data(epoch, model, optimizer, scheduler, batch_step, config, lock, device):
    if (epoch + 1) % 20 == 0:
        save_checkpoint(model, optimizer, scheduler, epoch, batch_step, config)
        torch.save(model.state_dict(), config['model_path'])
        # Validate only every 10 epochs
    if (epoch + 1) % 20 == 0:
        validate_model(model, device)
