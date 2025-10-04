"""
Quick Start Script for Conditional Minecraft GPT
=================================================

This script automates the complete setup and training process:
1. Split dataset into train/val/test
2. Start training
3. Monitor progress

Usage:
    python quickstart_conditional.py
    
    # Custom data path
    python quickstart_conditional.py --data_path "path/to/schems"
    
    # Quick test (fewer epochs)
    python quickstart_conditional.py --test_mode
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Step: {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully!")


def check_requirements():
    """Check if required packages are installed."""
    print("Checking requirements...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found!")
        print("Please install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    try:
        import mcschematic
    except ImportError:
        print("❌ mcschematic not found!")
        print("Please install: pip install mcschematic")
        sys.exit(1)
    
    print("✓ All requirements satisfied!")


def load_config():
    """Load configuration."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Quick start for conditional Minecraft GPT')
    parser.add_argument('--data_path', type=str, default='fixed_all_files (1)/fixed_all_files',
                        help='Path to .schem files')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode (fewer epochs, smaller batches)')
    parser.add_argument('--skip_split', action='store_true',
                        help='Skip dataset splitting (if already done)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Conditional Minecraft GPT - Quick Start")
    print("="*80)
    
    # Check requirements
    check_requirements()
    
    # Load config
    config = load_config()
    print(f"\nConfiguration loaded:")
    print(f"  Model: {config['model']['n_layers']} layers, {config['model']['d_model']} d_model")
    print(f"  Training: {config['training']['num_epochs']} epochs, batch size {config['training']['batch_size']}")
    
    # Adjust for test mode
    if args.test_mode:
        print("\n⚠️  Running in TEST MODE")
        args.epochs = args.epochs or 5
        args.batch_size = args.batch_size or 4
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
    else:
        # Use config values if not overridden
        args.epochs = args.epochs or config['training']['num_epochs']
        args.batch_size = args.batch_size or config['training']['batch_size']
    
    # Check if data exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"\n❌ Error: Data path not found: {data_path}")
        sys.exit(1)
    
    schem_files = list(data_path.glob('*.schem'))
    print(f"\n✓ Found {len(schem_files)} .schem files in {data_path}")
    
    # Step 1: Split dataset (if not skipped)
    if not args.skip_split:
        if not Path('data/train').exists():
            cmd = f'python src/prepare_dataset.py split --source "{args.data_path}" --output data'
            run_command(cmd, "Dataset Splitting")
        else:
            print("\n✓ Dataset already split (data/train exists)")
            print("  Use --skip_split to explicitly skip this step")
    else:
        print("\n⊘ Skipping dataset split")
    
    # Step 2: Build text vocabulary (if not exists)
    if not Path('text_vocab.json').exists():
        cmd = f'python src/text_tokenizer.py "{args.data_path}"'
        run_command(cmd, "Text Vocabulary Building")
    else:
        print("\n✓ Text vocabulary already exists (text_vocab.json)")
    
    # Step 3: Start training
    train_cmd = 'python src/conditional_train.py --data_path data/train --val_path data/val'
    
    if args.epochs is not None:
        train_cmd += f' --epochs {args.epochs}'
    if args.batch_size is not None:
        train_cmd += f' --batch_size {args.batch_size}'
    
    run_command(train_cmd, "Model Training")
    
    # Complete
    print("\n" + "="*80)
    print("Quick Start Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check training logs: tensorboard --logdir=runs")
    print("  2. Generate structures:")
    print('     python src/conditional_generate.py --prompt "a big medieval house" --checkpoint checkpoints/conditional_model_best.pt')
    print("  3. Monitor checkpoints in: checkpoints/")
    print("\nExample prompts to try:")
    print('  - "a big abandoned barn built out of spruce wood with interior"')
    print('  - "a medieval house with oak wood and stone base"')
    print('  - "a small church with pointed roof and stone walls"')
    print('  - "an arabic desert house out of sandstone"')
    print("="*80)


if __name__ == '__main__':
    main()
