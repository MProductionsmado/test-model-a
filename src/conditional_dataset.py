"""
Conditional Dataset for Text-to-Structure Generation
PyTorch Dataset that pairs text descriptions with structures
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import random

from schematic_parser import SchematicParser
from vocab import SPECIAL_TOKENS as BLOCK_SPECIAL_TOKENS, VOCAB_SIZE as BLOCK_VOCAB_SIZE
from text_tokenizer import TextTokenizer, TEXT_SPECIAL_TOKENS


class ConditionalStructureDataset(Dataset):
    """PyTorch Dataset for text-conditioned Minecraft structure generation.
    
    Supports two modes:
    1. Direct .schem loading (slower, more flexible)
    2. Pre-processed .npz loading (10-20x faster for training)
    
    To use fast mode, set preprocess=True on first load, then set use_cache=True.
    """
    
    def __init__(
        self,
        data_dir: str,
        text_tokenizer: TextTokenizer,
        target_size: Tuple[int, int, int] = (16, 16, 16),
        max_text_length: int = 128,
        augment: bool = True
    ):
        """
        Initialize the conditional dataset with automatic caching for optimal performance.
        
        Args:
            data_dir: Directory containing .schem files
            text_tokenizer: Initialized TextTokenizer instance
            target_size: Target structure dimensions (x, y, z)
            max_text_length: Maximum length for text descriptions
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.text_tokenizer = text_tokenizer
        self.target_size = target_size
        self.max_text_length = max_text_length
        self.augment = augment
        
        # Parser with automatic caching (OPTIMAL PERFORMANCE!)
        self.parser = SchematicParser(target_size, cache_parsed=True)
        
        # Find all .schem files
        self.file_paths = glob.glob(os.path.join(data_dir, "**/*.schem"), recursive=True)
        
        if len(self.file_paths) == 0:
            print(f"WARNING: No .schem files found in {data_dir}")
        else:
            print(f"Found {len(self.file_paths)} .schem files in {data_dir}")
        
        # Pre-process text descriptions
        self.text_cache = {}
        print("Pre-processing text descriptions...")
        for filepath in self.file_paths:
            description = self.text_tokenizer.extract_description_from_filename(filepath)
            text_ids = self.text_tokenizer.encode(description, max_length=max_text_length)
            text_ids_padded = self.text_tokenizer.pad_sequence(text_ids, max_text_length)
            self.text_cache[filepath] = text_ids_padded
        print("Text pre-processing complete!")
        
        # Pre-load all structures into cache (CRITICAL FOR SPEED!)
        print(f"Pre-loading {len(self.file_paths)} structures into memory cache...")
        from tqdm import tqdm
        failed = 0
        for filepath in tqdm(self.file_paths, desc="Caching structures"):
            structure = self.parser.parse_file(filepath)
            if structure is None:
                failed += 1
        if failed > 0:
            print(f"Warning: {failed} files failed to parse")
        print(f"✓ All structures cached! Training is now optimized!")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a (text, structure) pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (text_ids, text_mask, input_blocks, target_blocks)
        """
        filepath = self.file_paths[idx]
        
        # Get text from cache
        text_ids = self.text_cache[filepath]
        text_mask = [1 if tid != TEXT_SPECIAL_TOKENS['<PAD>'] else 0 for tid in text_ids]
        
        # Get structure from parser cache (already pre-loaded!)
        structure = self.parser.parse_file(filepath)
        
        if structure is None:
            # Return empty structure on error
            structure = np.full(
                np.prod(self.target_size),
                BLOCK_SPECIAL_TOKENS['<AIR>'],
                dtype=np.int32
            )
        
        # Apply augmentation
        if self.augment:
            structure = self._augment_structure(structure)
        
        # Create input and target sequences for autoregressive training
        # Input:  [<START>, block1, block2, ..., blockN]  (length = N+1)
        # Target: [block1, block2, ..., blockN, <END>]    (length = N+1)
        # This way the model predicts the next block at each position
        
        input_blocks = np.concatenate([
            [BLOCK_SPECIAL_TOKENS['<START>']],
            structure
        ])
        
        target_blocks = np.concatenate([
            structure,
            [BLOCK_SPECIAL_TOKENS['<END>']]
        ])
        
        # Convert to tensors
        text_ids_tensor = torch.from_numpy(np.array(text_ids)).long()
        text_mask_tensor = torch.from_numpy(np.array(text_mask)).long()
        input_blocks_tensor = torch.from_numpy(input_blocks).long()
        target_blocks_tensor = torch.from_numpy(target_blocks).long()
        
        return {
            'text_ids': text_ids_tensor,
            'text_mask': text_mask_tensor,
            'input_blocks': input_blocks_tensor,
            'target_blocks': target_blocks_tensor
        }
    
    def _augment_structure(self, structure: np.ndarray) -> np.ndarray:
        """Apply data augmentation to structure."""
        # Unflatten to 3D
        structure_3d = self.parser.unflatten_structure(structure)
        
        # Random 90° rotations around Y axis
        if random.random() > 0.5:
            k = random.randint(1, 3)
            structure_3d = np.rot90(structure_3d, k=k, axes=(0, 2))
        
        # Random flipping
        if random.random() > 0.5:
            structure_3d = np.flip(structure_3d, axis=0)  # Flip X
        if random.random() > 0.5:
            structure_3d = np.flip(structure_3d, axis=2)  # Flip Z
        
        # Flatten back
        return self.parser.flatten_structure(structure_3d)
    
    def get_text_description(self, idx: int) -> str:
        """Get the text description for a sample."""
        filepath = self.file_paths[idx]
        description = self.text_tokenizer.extract_description_from_filename(filepath)
        return description
    
    def get_structure_3d(self, idx: int) -> np.ndarray:
        """Get structure as 3D array."""
        filepath = self.file_paths[idx]
        structure_flat = self.parser.parse_file(filepath)
        if structure_flat is None:
            return np.full(self.target_size, BLOCK_SPECIAL_TOKENS['<AIR>'], dtype=np.int32)
        return self.parser.unflatten_structure(structure_flat)


def create_conditional_dataloaders(
    train_dir: str,
    val_dir: str,
    text_vocab_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    target_size: Tuple[int, int, int] = (16, 16, 16),
    max_text_length: int = 128
) -> Tuple[DataLoader, DataLoader, TextTokenizer]:
    """
    Create conditional training and validation dataloaders.
    
    Args:
        train_dir: Directory with training .schem files
        val_dir: Directory with validation .schem files
        text_vocab_path: Path to text vocabulary JSON file
        batch_size: Batch size for training
        num_workers: Number of worker processes
        target_size: Target structure dimensions
        max_text_length: Maximum text description length
        
    Returns:
        Tuple of (train_loader, val_loader, text_tokenizer)
    """
    # Load text tokenizer
    print(f"Loading text vocabulary from {text_vocab_path}...")
    text_tokenizer = TextTokenizer()
    text_tokenizer.load_vocab(text_vocab_path)
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = ConditionalStructureDataset(
        train_dir,
        text_tokenizer=text_tokenizer,
        target_size=target_size,
        max_text_length=max_text_length,
        augment=True,
        cache=False
    )
    
    print("\nCreating validation dataset...")
    val_dataset = ConditionalStructureDataset(
        val_dir,
        text_tokenizer=text_tokenizer,
        target_size=target_size,
        max_text_length=max_text_length,
        augment=False,
        cache=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, text_tokenizer


if __name__ == "__main__":
    print("Testing ConditionalStructureDataset...")
    
    # Load text tokenizer
    text_tokenizer = TextTokenizer()
    
    # Try to load vocabulary
    try:
        text_tokenizer.load_vocab("text_vocab.json")
        
        # Create dataset
        dataset = ConditionalStructureDataset(
            data_dir="fixed_all_files (1)/fixed_all_files",
            text_tokenizer=text_tokenizer,
            target_size=(16, 16, 16),
            max_text_length=128,
            augment=False
        )
        
        if len(dataset) > 0:
            # Get a sample
            text_ids, text_mask, input_blocks, target_blocks = dataset[0]
            
            print(f"\nSample 0:")
            print(f"  Text description: {dataset.get_text_description(0)}")
            print(f"  Text IDs shape: {text_ids.shape}")
            print(f"  Text mask sum: {text_mask.sum()}")  # Number of non-padding tokens
            print(f"  Input blocks shape: {input_blocks.shape}")
            print(f"  Target blocks shape: {target_blocks.shape}")
            
            # Decode text
            decoded_text = text_tokenizer.decode(text_ids.tolist())
            print(f"  Decoded text: {decoded_text}")
            
            print("\n✓ Dataset test successful!")
        else:
            print("Dataset is empty")
    
    except FileNotFoundError:
        print("text_vocab.json not found. Run text_tokenizer.py first to build vocabulary.")
