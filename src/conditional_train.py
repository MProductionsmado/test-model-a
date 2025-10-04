"""
Conditional Training Script for Text-to-Structure Minecraft GPT
================================================================

This script trains the conditional GPT model that generates Minecraft structures
based on text descriptions. It uses cross-attention to condition the generation
on the input text prompt.

Usage:
    python src/conditional_train.py --data_path "data/train" --epochs 100
    
    # Resume training from checkpoint
    python src/conditional_train.py --data_path "data/train" --resume checkpoints/conditional_model_epoch_50.pt
    
    # Custom settings
    python src/conditional_train.py --data_path "data/train" --batch_size 8 --lr 0.0001
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

from conditional_model import ConditionalMinecraftGPT
from conditional_dataset import ConditionalStructureDataset
from text_tokenizer import TextTokenizer
from vocab import get_block_vocab


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, block_vocab_size, text_vocab_size, device):
    """Create conditional GPT model."""
    model = ConditionalMinecraftGPT(
        block_vocab_size=block_vocab_size,
        text_vocab_size=text_vocab_size,
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        max_seq_length=config['model']['max_seq_length'],
        max_text_length=config['conditional']['max_text_length'],
        text_encoder_layers=config['conditional']['text_encoder_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    return model


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        text_ids = batch['text_ids'].to(device)
        text_mask = batch['text_mask'].to(device)
        input_blocks = batch['input_blocks'].to(device)
        target_blocks = batch['target_blocks'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_blocks, text_ids, text_mask)
        
        # Calculate loss
        # logits: (batch_size, seq_length, vocab_size)
        # target_blocks: (batch_size, seq_length)
        loss = criterion(logits.view(-1, logits.size(-1)), target_blocks.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        
        global_step += 1
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, global_step


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            input_blocks = batch['input_blocks'].to(device)
            target_blocks = batch['target_blocks'].to(device)
            
            # Forward pass
            logits = model(input_blocks, text_ids, text_mask)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_blocks.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Checkpoint loaded: {filepath}")
    print(f"  Resuming from epoch {epoch} (loss: {loss:.4f})")
    
    return epoch, loss


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Train conditional Minecraft structure GPT')
    parser.add_argument('--data_path', type=str, default='data/train',
                        help='Path to training .schem files')
    parser.add_argument('--val_path', type=str, default='data/val',
                        help='Path to validation .schem files')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--text_vocab', type=str, default='text_vocab.json',
                        help='Path to text vocabulary file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable tensorboard logging')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Normalize config: support both 'epochs' and 'num_epochs'
    if 'num_epochs' in config['training'] and 'epochs' not in config['training']:
        config['training']['epochs'] = config['training']['num_epochs']
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    block_vocab = get_block_vocab()
    block_vocab_size = len(block_vocab)
    print(f"Block vocabulary size: {block_vocab_size}")
    
    text_tokenizer = TextTokenizer.load(args.text_vocab)
    text_vocab_size = text_tokenizer.vocab_size
    print(f"Text vocabulary size: {text_vocab_size}")
    
    # Create datasets
    print("\nLoading datasets...")
    target_size = (
        config['structure']['size_x'],
        config['structure']['size_y'],
        config['structure']['size_z']
    )
    train_dataset = ConditionalStructureDataset(
        data_dir=args.data_path,
        text_tokenizer=text_tokenizer,
        target_size=target_size,
        max_text_length=config['conditional']['max_text_length'],
        augment=config['training']['augment']
    )
    print(f"Training samples: {len(train_dataset)}")
    
    val_dataset = None
    if Path(args.val_path).exists():
        val_dataset = ConditionalStructureDataset(
            data_dir=args.val_path,
            text_tokenizer=text_tokenizer,
            target_size=target_size,
            max_text_length=config['conditional']['max_text_length'],
            augment=False  # No augmentation for validation
        )
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True if device.type == 'cuda' else False
        )
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, block_vocab_size, text_vocab_size, device)
    
    # Enable gradient checkpointing if specified
    if config['training'].get('gradient_checkpointing', False):
        model.use_gradient_checkpointing = True
        print("✓ Gradient checkpointing enabled (saves memory)")
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=block_vocab['<PAD>'])
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume is not None:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1  # Start from next epoch
    
    # Setup tensorboard
    writer = None
    if not args.no_tensorboard:
        log_dir = Path('runs') / f'conditional_train_{time.strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(log_dir)
        print(f"\nTensorboard logs: {log_dir}")
        print(f"Run: tensorboard --logdir=runs")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()
        
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, writer, global_step
        )
        
        # Validate
        val_loss = None
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            if val_loss is not None:
                writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Train/Epoch_Time', epoch_time, epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'conditional_model_epoch_{epoch}.pt'
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
        
        # Save best model
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / 'conditional_model_best.pt'
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"  ✓ New best model! Val loss: {val_loss:.4f}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'conditional_model_final.pt'
    save_checkpoint(model, optimizer, config['training']['epochs'] - 1, train_loss, final_model_path)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    if val_loss is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved: {final_model_path}")
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
