"""
Minecraft Block Vocabulary
Comprehensive list of Minecraft blocks with their IDs for the AI model
"""

# Special tokens
SPECIAL_TOKENS = {
    '<PAD>': 0,      # Padding token
    '<UNK>': 1,      # Unknown token
    '<START>': 2,    # Start of sequence
    '<END>': 3,      # End of sequence
    '<AIR>': 4,      # Air block (empty space)
}

# Minecraft blocks (75+ blocks)
MINECRAFT_BLOCKS = {
    # Natural Blocks
    'minecraft:stone': 5,
    'minecraft:granite': 6,
    'minecraft:polished_granite': 7,
    'minecraft:diorite': 8,
    'minecraft:polished_diorite': 9,
    'minecraft:andesite': 10,
    'minecraft:polished_andesite': 11,
    'minecraft:grass_block': 12,
    'minecraft:dirt': 13,
    'minecraft:coarse_dirt': 14,
    'minecraft:podzol': 15,
    'minecraft:cobblestone': 16,
    'minecraft:oak_log': 17,
    'minecraft:spruce_log': 18,
    'minecraft:birch_log': 19,
    'minecraft:jungle_log': 20,
    'minecraft:acacia_log': 21,
    'minecraft:dark_oak_log': 22,
    'minecraft:oak_planks': 23,
    'minecraft:spruce_planks': 24,
    'minecraft:birch_planks': 25,
    'minecraft:jungle_planks': 26,
    'minecraft:acacia_planks': 27,
    'minecraft:dark_oak_planks': 28,
    'minecraft:sand': 29,
    'minecraft:red_sand': 30,
    'minecraft:gravel': 31,
    'minecraft:gold_ore': 32,
    'minecraft:iron_ore': 33,
    'minecraft:coal_ore': 34,
    'minecraft:oak_leaves': 35,
    'minecraft:spruce_leaves': 36,
    'minecraft:birch_leaves': 37,
    'minecraft:jungle_leaves': 38,
    
    # Building Blocks
    'minecraft:glass': 39,
    'minecraft:white_stained_glass': 40,
    'minecraft:stone_bricks': 41,
    'minecraft:mossy_stone_bricks': 42,
    'minecraft:cracked_stone_bricks': 43,
    'minecraft:chiseled_stone_bricks': 44,
    'minecraft:bricks': 45,
    'minecraft:smooth_stone': 46,
    'minecraft:sandstone': 47,
    'minecraft:smooth_sandstone': 48,
    'minecraft:chiseled_sandstone': 49,
    'minecraft:red_sandstone': 50,
    'minecraft:quartz_block': 51,
    'minecraft:quartz_pillar': 52,
    'minecraft:chiseled_quartz_block': 53,
    'minecraft:smooth_quartz': 54,
    'minecraft:purpur_block': 55,
    'minecraft:purpur_pillar': 56,
    'minecraft:prismarine': 57,
    'minecraft:prismarine_bricks': 58,
    'minecraft:dark_prismarine': 59,
    
    # Functional Blocks
    'minecraft:oak_stairs': 60,
    'minecraft:stone_stairs': 61,
    'minecraft:brick_stairs': 62,
    'minecraft:stone_brick_stairs': 63,
    'minecraft:oak_slab': 64,
    'minecraft:stone_slab': 65,
    'minecraft:smooth_stone_slab': 66,
    'minecraft:oak_fence': 67,
    'minecraft:oak_door': 68,
    'minecraft:iron_door': 69,
    'minecraft:oak_trapdoor': 70,
    'minecraft:iron_trapdoor': 71,
    'minecraft:glass_pane': 72,
    'minecraft:iron_bars': 73,
    'minecraft:chain': 74,
    'minecraft:lantern': 75,
    'minecraft:torch': 76,
    
    # Decorative Blocks
    'minecraft:white_wool': 77,
    'minecraft:orange_wool': 78,
    'minecraft:magenta_wool': 79,
    'minecraft:light_blue_wool': 80,
    'minecraft:yellow_wool': 81,
    'minecraft:lime_wool': 82,
    'minecraft:pink_wool': 83,
    'minecraft:gray_wool': 84,
    'minecraft:light_gray_wool': 85,
    'minecraft:cyan_wool': 86,
    'minecraft:purple_wool': 87,
    'minecraft:blue_wool': 88,
    'minecraft:brown_wool': 89,
    'minecraft:green_wool': 90,
    'minecraft:red_wool': 91,
    'minecraft:black_wool': 92,
    'minecraft:white_concrete': 93,
    'minecraft:orange_concrete': 94,
    'minecraft:magenta_concrete': 95,
    'minecraft:light_blue_concrete': 96,
    'minecraft:yellow_concrete': 97,
    'minecraft:lime_concrete': 98,
    'minecraft:pink_concrete': 99,
    'minecraft:gray_concrete': 100,
    'minecraft:light_gray_concrete': 101,
    'minecraft:cyan_concrete': 102,
    'minecraft:purple_concrete': 103,
    'minecraft:blue_concrete': 104,
    'minecraft:brown_concrete': 105,
    'minecraft:green_concrete': 106,
    'minecraft:red_concrete': 107,
    'minecraft:black_concrete': 108,
    
    # Additional Building Materials
    'minecraft:terracotta': 109,
    'minecraft:white_terracotta': 110,
    'minecraft:glowstone': 111,
    'minecraft:sea_lantern': 112,
    'minecraft:obsidian': 113,
    'minecraft:netherrack': 114,
    'minecraft:nether_bricks': 115,
    'minecraft:end_stone': 116,
    'minecraft:end_stone_bricks': 117,
    'minecraft:bedrock': 118,
    'minecraft:water': 119,
    'minecraft:lava': 120,
}

# Combine all tokens
BLOCK_VOCAB = {**SPECIAL_TOKENS, **MINECRAFT_BLOCKS}

# Reverse mapping
ID_TO_BLOCK = {v: k for k, v in BLOCK_VOCAB.items()}

# Vocabulary size
VOCAB_SIZE = len(BLOCK_VOCAB)

# Block properties for learning
BLOCK_PROPERTIES = {
    'transparent': ['<AIR>', 'minecraft:glass', 'minecraft:white_stained_glass', 'minecraft:glass_pane', 'minecraft:water'],
    'solid': ['minecraft:stone', 'minecraft:cobblestone', 'minecraft:bricks', 'minecraft:planks'],
    'natural': ['minecraft:stone', 'minecraft:dirt', 'minecraft:grass_block', 'minecraft:oak_log', 'minecraft:oak_leaves'],
    'wooden': ['minecraft:oak_planks', 'minecraft:spruce_planks', 'minecraft:birch_planks', 'minecraft:oak_log'],
    'decorative': ['minecraft:white_wool', 'minecraft:white_concrete', 'minecraft:terracotta'],
}

def get_block_id(block_name: str) -> int:
    """Get the ID for a block name."""
    return BLOCK_VOCAB.get(block_name, SPECIAL_TOKENS['<UNK>'])

def get_block_name(block_id: int) -> str:
    """Get the block name for an ID."""
    return ID_TO_BLOCK.get(block_id, '<UNK>')

def is_special_token(block_id: int) -> bool:
    """Check if a block ID is a special token."""
    return block_id in SPECIAL_TOKENS.values()

def get_block_vocab():
    """Return the complete block vocabulary dictionary."""
    return BLOCK_VOCAB.copy()

def print_vocab_stats():
    """Print vocabulary statistics."""
    print(f"Total vocabulary size: {VOCAB_SIZE}")
    print(f"Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"Minecraft blocks: {len(MINECRAFT_BLOCKS)}")
    print("\nBlock categories:")
    for category, blocks in BLOCK_PROPERTIES.items():
        print(f"  {category}: {len(blocks)} blocks")

if __name__ == "__main__":
    print_vocab_stats()
    print("\n=== Sample Block Mappings ===")
    samples = ['minecraft:stone', 'minecraft:oak_planks', 'minecraft:glass', '<AIR>']
    for block in samples:
        block_id = get_block_id(block)
        print(f"{block:30} -> ID {block_id:3}")
