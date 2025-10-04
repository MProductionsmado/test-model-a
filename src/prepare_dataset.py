"""
Prepare Dataset Script
Helper script to organize and validate .schem files for training
"""

import os
import shutil
import glob
import argparse
from typing import List
import random

from schematic_parser import SchematicParser


def find_schem_files(directory: str) -> List[str]:
    """Find all .schem files in a directory."""
    return glob.glob(os.path.join(directory, "**/*.schem"), recursive=True)


def validate_schematic(filepath: str, parser: SchematicParser) -> bool:
    """
    Validate a schematic file.
    
    Args:
        filepath: Path to .schem file
        parser: SchematicParser instance
        
    Returns:
        True if valid, False otherwise
    """
    try:
        structure = parser.parse_file(filepath)
        if structure is None:
            print(f"  ✗ {filepath} - Failed to parse")
            return False
        
        # Check if structure is not empty
        from vocab import SPECIAL_TOKENS
        import numpy as np
        
        non_air = np.sum(structure != SPECIAL_TOKENS['<AIR>'])
        if non_air == 0:
            print(f"  ✗ {filepath} - Empty structure (all air)")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ {filepath} - Error: {e}")
        return False


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        source_dir: Directory containing .schem files
        output_dir: Output directory for split datasets
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Find all files
    print(f"Searching for .schem files in {source_dir}...")
    files = find_schem_files(source_dir)
    print(f"Found {len(files)} files")
    
    if len(files) == 0:
        print("No files found!")
        return
    
    # Validate files
    print("\nValidating files...")
    parser = SchematicParser()
    valid_files = []
    
    for filepath in files:
        if validate_schematic(filepath, parser):
            valid_files.append(filepath)
    
    print(f"\n{len(valid_files)}/{len(files)} files are valid")
    
    if len(valid_files) == 0:
        print("No valid files to split!")
        return
    
    # Shuffle files
    random.seed(seed)
    random.shuffle(valid_files)
    
    # Calculate split indices
    n_total = len(valid_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train + n_val]
    test_files = valid_files[n_train + n_val:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy files
    print("\nCopying files...")
    
    def copy_files(files, dest_dir, label):
        for i, src in enumerate(files):
            filename = os.path.basename(src)
            dest = os.path.join(dest_dir, filename)
            shutil.copy2(src, dest)
            if (i + 1) % 10 == 0:
                print(f"  {label}: {i + 1}/{len(files)}")
    
    copy_files(train_files, train_dir, "Train")
    copy_files(val_files, val_dir, "Val")
    copy_files(test_files, test_dir, "Test")
    
    print("\n✓ Dataset preparation complete!")
    print(f"  Output directory: {output_dir}")


def analyze_directory(directory: str):
    """Analyze a directory of .schem files."""
    from utils import analyze_dataset_statistics
    
    print(f"\nAnalyzing {directory}...")
    stats = analyze_dataset_statistics(directory)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for MinecraftGPT training"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory with .schem files'
    )
    split_parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory (default: data)'
    )
    split_parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    split_parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    split_parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )
    split_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Directory to analyze'
    )
    analyze_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for statistics'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate .schem files')
    validate_parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Directory with .schem files'
    )
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    
    elif args.command == 'analyze':
        analyze_directory(args.directory)
    
    elif args.command == 'validate':
        files = find_schem_files(args.directory)
        print(f"Found {len(files)} files")
        print("\nValidating...")
        
        parser_obj = SchematicParser()
        valid_count = 0
        
        for filepath in files:
            if validate_schematic(filepath, parser_obj):
                valid_count += 1
        
        print(f"\n{valid_count}/{len(files)} files are valid")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
