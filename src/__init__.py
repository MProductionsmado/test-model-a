"""
MinecraftGPT Package
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "MinecraftGPT Team"
__description__ = "GPT-based AI for generating Minecraft structures"

from .vocab import VOCAB_SIZE, get_block_id, get_block_name
from .model import MinecraftGPT
from .schematic_parser import SchematicParser
from .dataset import MinecraftStructureDataset, create_dataloaders

__all__ = [
    'VOCAB_SIZE',
    'get_block_id',
    'get_block_name',
    'MinecraftGPT',
    'SchematicParser',
    'MinecraftStructureDataset',
    'create_dataloaders'
]
