"""
Schematic File Parser
Handles loading and parsing of .schem files using direct NBT parsing
Supports Sponge Schematic Format Version 2
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict
import nbtlib
from pathlib import Path

from vocab import BLOCK_VOCAB, SPECIAL_TOKENS


def get_block_id(block_name: str) -> int:
    """Get block ID from name, or return AIR if unknown."""
    # Remove minecraft: prefix if present
    if block_name.startswith('minecraft:'):
        block_name = block_name[10:]
    
    # Try to find in vocab
    full_name = f'minecraft:{block_name}'
    if full_name in BLOCK_VOCAB:
        return BLOCK_VOCAB[full_name]
    
    # Try without properties (e.g., "oak_planks[...]" -> "oak_planks")
    if '[' in block_name:
        block_name = block_name.split('[')[0]
        full_name = f'minecraft:{block_name}'
        if full_name in BLOCK_VOCAB:
            return BLOCK_VOCAB[full_name]
    
    # Default to air
    return SPECIAL_TOKENS['<PAD>']


def get_block_name(block_id: int) -> str:
    """Get block name from ID."""
    id_to_name = {v: k for k, v in BLOCK_VOCAB.items()}
    return id_to_name.get(block_id, 'minecraft:air')


class SchematicParser:
    """Parser for Minecraft .schem files (Sponge Format V2)."""
    
    def __init__(self, target_size: Tuple[int, int, int] = (16, 16, 16)):
        """
        Initialize the schematic parser.
        
        Args:
            target_size: Target dimensions (x, y, z) for structures
        """
        self.target_size = target_size
        self.size_x, self.size_y, self.size_z = target_size
        
    def load_schematic(self, filepath: str) -> Optional[Dict]:
        """
        Load a .schem file using NBT.
        
        Args:
            filepath: Path to the .schem file
            
        Returns:
            Dictionary with schematic data or None if loading fails
        """
        try:
            # Load NBT file
            nbt_file = nbtlib.load(filepath)
            
            # Extract schematic data
            if 'Schematic' in nbt_file:
                schem_data = nbt_file['Schematic']
            elif '' in nbt_file:  # Root compound
                schem_data = nbt_file['']
            else:
                schem_data = nbt_file
            
            # Extract dimensions
            width = int(schem_data.get('Width', 0))
            height = int(schem_data.get('Height', 0))
            length = int(schem_data.get('Length', 0))
            
            # Extract Blocks compound (contains Palette and Data)
            blocks_compound = schem_data.get('Blocks', None)
            
            if blocks_compound is None:
                print(f"No Blocks compound found in {filepath}")
                return None
            
            # Extract palette and data from Blocks compound
            palette = blocks_compound.get('Palette', None)
            data = blocks_compound.get('Data', None)
            
            if palette is None or data is None:
                print(f"No Palette or Data found in {filepath}")
                return None
            
            return {
                'width': width,
                'height': height,
                'length': length,
                'palette': palette,
                'data': data,
                'schematic': schem_data
            }
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def schematic_to_array(self, schem_data: Dict) -> np.ndarray:
        """
        Convert schematic to numpy array of block IDs.
        
        Convert schematic data to numpy array of block IDs.
        
        Args:
            schem_data: Dictionary with schematic data
            
        Returns:
            3D numpy array of block IDs
        """
        width = schem_data['width']
        height = schem_data['height']
        length = schem_data['length']
        palette = schem_data['palette']
        data = schem_data['data']
        
        # Create array
        block_array = np.zeros((width, height, length), dtype=np.int32)
        
        try:
            # Build palette mapping (palette_id -> our_block_id)
            palette_mapping = np.full(256, SPECIAL_TOKENS['<PAD>'], dtype=np.int32)  # Support up to 256 palette entries
            
            for block_name_key in palette.keys():
                block_name = str(block_name_key)
                palette_id = int(palette[block_name_key])
                if palette_id < 256:
                    palette_mapping[palette_id] = get_block_id(block_name)
            
            # Convert Data array to numpy efficiently
            if isinstance(data, np.ndarray):
                block_indices = data.astype(np.int32)
            else:
                # ByteArray from NBT
                block_indices = np.frombuffer(bytes(data), dtype=np.uint8).astype(np.int32)
            
            # Calculate expected size
            expected_size = width * height * length
            
            if len(block_indices) >= expected_size:
                # Truncate to expected size
                block_indices = block_indices[:expected_size]
                
                # Map through palette using vectorized operation
                block_ids_flat = palette_mapping[block_indices]
                
                # Reshape to 3D (Y-Z-X ordering, then transpose to X-Y-Z)
                block_array = block_ids_flat.reshape(height, length, width).transpose(2, 0, 1)
            else:
                # Not enough data, fill partially
                print(f"Warning: Expected {expected_size} blocks, got {len(block_indices)}")
                block_array[:] = SPECIAL_TOKENS['<PAD>']
                
        except Exception as e:
            print(f"Error parsing blocks: {e}")
            # Return air-filled array
            block_array = np.full((width, height, length), SPECIAL_TOKENS['<PAD>'], dtype=np.int32)
        
        return block_array
    
    def resize_structure(self, block_array: np.ndarray) -> np.ndarray:
        """
        Resize structure to target dimensions.
        Crops if larger, pads with air if smaller.
        
        Args:
            block_array: Input 3D array
            
        Returns:
            Resized 3D array
        """
        curr_x, curr_y, curr_z = block_array.shape
        target_x, target_y, target_z = self.target_size
        
        # Create output array filled with air
        output = np.full(self.target_size, SPECIAL_TOKENS['<AIR>'], dtype=np.int32)
        
        # Calculate how much to copy
        copy_x = min(curr_x, target_x)
        copy_y = min(curr_y, target_y)
        copy_z = min(curr_z, target_z)
        
        # Copy the data (center it if smaller)
        offset_x = (target_x - copy_x) // 2
        offset_y = 0  # Start from bottom
        offset_z = (target_z - copy_z) // 2
        
        output[offset_x:offset_x+copy_x, 
               offset_y:offset_y+copy_y, 
               offset_z:offset_z+copy_z] = block_array[:copy_x, :copy_y, :copy_z]
        
        return output
    
    def flatten_structure(self, block_array: np.ndarray) -> np.ndarray:
        """
        Flatten 3D structure to 1D sequence.
        Uses Y-Z-X ordering (height first, then depth, then width).
        
        Args:
            block_array: 3D array of block IDs
            
        Returns:
            1D array of block IDs
        """
        # Transpose to Y-Z-X ordering and flatten
        return block_array.transpose(1, 2, 0).flatten()
    
    def unflatten_structure(self, flat_array: np.ndarray) -> np.ndarray:
        """
        Convert 1D sequence back to 3D structure.
        
        Args:
            flat_array: 1D array of block IDs
            
        Returns:
            3D array of block IDs
        """
        # Reshape to Y-Z-X and transpose back to X-Y-Z
        y, z, x = self.target_size[1], self.target_size[2], self.target_size[0]
        return flat_array.reshape(y, z, x).transpose(2, 0, 1)
    
    def parse_file(self, filepath: str) -> Optional[np.ndarray]:
        """
        Complete pipeline: load, parse, resize, and flatten a .schem file.
        
        Args:
            filepath: Path to .schem file
            
        Returns:
            1D numpy array of block IDs, or None on failure
        """
        schem = self.load_schematic(filepath)
        if schem is None:
            return None
        
        block_array = self.schematic_to_array(schem)
        resized = self.resize_structure(block_array)
        flattened = self.flatten_structure(resized)
        
        return flattened
    
    def array_to_schematic(self, block_array: np.ndarray, output_path: str, description: str = "") -> bool:
        """
        Convert block array to .schem file.
        
        Args:
            block_array: 3D array of block IDs
            output_path: Path to save .schem file
            description: Optional description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            x, y, z = block_array.shape
            
            # Create palette and block data
            palette = {}
            blocks_list = []
            palette_counter = 0
            
            # Build palette and blocks
            for iy in range(y):
                for iz in range(z):
                    for ix in range(x):
                        block_id = block_array[ix, iy, iz]
                        block_name = get_block_name(block_id)
                        
                        # Add to palette if new
                        if block_name not in palette:
                            palette[block_name] = palette_counter
                            palette_counter += 1
                        
                        blocks_list.append(palette[block_name])
            
            # Create NBT structure
            schem_nbt = nbtlib.Compound({
                'Version': nbtlib.Int(2),
                'DataVersion': nbtlib.Int(3218),  # Minecraft 1.19
                'Width': nbtlib.Short(x),
                'Height': nbtlib.Short(y),
                'Length': nbtlib.Short(z),
                'Palette': nbtlib.Compound({
                    block_name: nbtlib.Int(block_id)
                    for block_name, block_id in palette.items()
                }),
                'BlockData': nbtlib.ByteArray(blocks_list),
                'Metadata': nbtlib.Compound({
                    'Date': nbtlib.Long(0),
                    'Name': nbtlib.String(description)
                })
            })
            
            # Save file
            nbt_file = nbtlib.File({'': schem_nbt})
            nbt_file.save(output_path, gzipped=True)
            
            return True
            
        except Exception as e:
            print(f"Error creating schematic: {e}")
            return False
    
    def get_structure_stats(self, block_array: np.ndarray) -> Dict:
        """
        Get statistics about a structure.
        
        Args:
            block_array: 3D or 1D array of block IDs
            
        Returns:
            Dictionary with statistics
        """
        unique, counts = np.unique(block_array, return_counts=True)
        
        total_blocks = len(block_array.flatten())
        air_blocks = counts[unique == SPECIAL_TOKENS['<AIR>']][0] if SPECIAL_TOKENS['<AIR>'] in unique else 0
        solid_blocks = total_blocks - air_blocks
        
        stats = {
            'total_blocks': total_blocks,
            'solid_blocks': solid_blocks,
            'air_blocks': air_blocks,
            'unique_blocks': len(unique),
            'density': solid_blocks / total_blocks,
            'block_distribution': dict(zip(unique.tolist(), counts.tolist()))
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    parser = SchematicParser(target_size=(16, 16, 16))
    
    print("Schematic Parser initialized")
    print(f"Target size: {parser.target_size}")
    print(f"Total sequence length: {np.prod(parser.target_size)}")
