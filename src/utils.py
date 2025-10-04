"""
Utility Functions
Helper functions for visualization, analysis, and debugging
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import List, Dict, Optional
import json

from vocab import get_block_name, SPECIAL_TOKENS, BLOCK_PROPERTIES


def visualize_structure_3d(
    structure_3d: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Minecraft Structure",
    show_air: bool = False
):
    """
    Visualize a 3D structure using matplotlib.
    
    Args:
        structure_3d: 3D array of block IDs
        output_path: Optional path to save the figure
        title: Plot title
        show_air: Whether to show air blocks
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x_size, y_size, z_size = structure_3d.shape
    
    # Get block positions and types
    xs, ys, zs = [], [], []
    colors = []
    
    # Color map for different block types
    block_colors = {
        'stone': '#808080',
        'wood': '#8B4513',
        'planks': '#DEB887',
        'grass': '#7CFC00',
        'dirt': '#8B4513',
        'glass': '#87CEEB',
        'wool': '#FFFFFF',
        'concrete': '#A9A9A9',
        'bricks': '#B22222',
        'leaves': '#228B22',
        'default': '#696969'
    }
    
    for x in range(x_size):
        for y in range(y_size):
            for z in range(z_size):
                block_id = structure_3d[x, y, z]
                
                # Skip air blocks unless requested
                if not show_air and block_id == SPECIAL_TOKENS['<AIR>']:
                    continue
                
                # Skip other special tokens
                block_name = get_block_name(block_id)
                if block_name.startswith('<') and block_name != '<AIR>':
                    continue
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
                
                # Determine color based on block type
                color = block_colors['default']
                for key, col in block_colors.items():
                    if key in block_name.lower():
                        color = col
                        break
                colors.append(color)
    
    # Plot voxels
    if len(xs) > 0:
        ax.scatter(xs, ys, zs, c=colors, marker='s', s=20, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Height)')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = max(x_size, y_size, z_size)
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_structure_2d_slices(
    structure_3d: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Structure Slices"
):
    """
    Visualize horizontal slices of a 3D structure.
    
    Args:
        structure_3d: 3D array of block IDs
        output_path: Optional path to save the figure
        title: Plot title
    """
    x_size, y_size, z_size = structure_3d.shape
    
    # Create subplots for each Y level
    n_cols = 4
    n_rows = (y_size + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    air_id = SPECIAL_TOKENS['<AIR>']
    
    for y in range(y_size):
        ax = axes[y]
        
        # Get slice at this height
        slice_2d = structure_3d[:, y, :]
        
        # Replace air with a special value for visualization
        vis_slice = slice_2d.copy().astype(float)
        vis_slice[vis_slice == air_id] = np.nan
        
        # Plot
        im = ax.imshow(vis_slice.T, cmap='tab20', interpolation='nearest', origin='lower')
        ax.set_title(f'Y={y}')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(y_size, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, y=1.0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Slice visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_dataset_statistics(
    data_dir: str,
    output_path: Optional[str] = None
) -> Dict:
    """
    Analyze statistics of a dataset directory.
    
    Args:
        data_dir: Directory containing .schem files
        output_path: Optional path to save statistics JSON
        
    Returns:
        Dictionary with dataset statistics
    """
    from schematic_parser import SchematicParser
    import glob
    
    parser = SchematicParser()
    file_paths = glob.glob(os.path.join(data_dir, "**/*.schem"), recursive=True)
    
    if len(file_paths) == 0:
        print(f"No .schem files found in {data_dir}")
        return {}
    
    print(f"Analyzing {len(file_paths)} files...")
    
    all_block_counts = {}
    densities = []
    sizes = []
    
    for filepath in file_paths:
        structure = parser.parse_file(filepath)
        if structure is None:
            continue
        
        stats = parser.get_structure_stats(structure)
        densities.append(stats['density'])
        sizes.append(stats['solid_blocks'])
        
        # Aggregate block counts
        for block_id, count in stats['block_distribution'].items():
            if block_id not in all_block_counts:
                all_block_counts[block_id] = 0
            all_block_counts[block_id] += count
    
    # Compute statistics
    total_blocks = sum(all_block_counts.values())
    air_blocks = all_block_counts.get(SPECIAL_TOKENS['<AIR>'], 0)
    
    # Block frequency (excluding air)
    block_frequencies = {
        get_block_name(bid): count / (total_blocks - air_blocks)
        for bid, count in all_block_counts.items()
        if bid != SPECIAL_TOKENS['<AIR>']
    }
    
    # Sort by frequency
    sorted_blocks = sorted(
        block_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    statistics = {
        'num_files': len(file_paths),
        'total_blocks': int(total_blocks),
        'air_blocks': int(air_blocks),
        'solid_blocks': int(total_blocks - air_blocks),
        'avg_density': float(np.mean(densities)),
        'std_density': float(np.std(densities)),
        'avg_size': float(np.mean(sizes)),
        'unique_blocks': len(all_block_counts) - 1,  # Exclude air
        'top_10_blocks': sorted_blocks[:10]
    }
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    print(f"Number of structures: {statistics['num_files']}")
    print(f"Total blocks: {statistics['total_blocks']:,}")
    print(f"Solid blocks: {statistics['solid_blocks']:,}")
    print(f"Average density: {statistics['avg_density']:.2%}")
    print(f"Unique block types: {statistics['unique_blocks']}")
    print("\nTop 10 most common blocks:")
    for i, (block, freq) in enumerate(sorted_blocks[:10], 1):
        print(f"  {i}. {block:40} {freq:.2%}")
    print("="*50)
    
    # Save to file
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        print(f"\nStatistics saved to: {output_path}")
    
    return statistics


def plot_training_curves(
    log_dir: str,
    output_path: Optional[str] = None
):
    """
    Plot training curves from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        output_path: Optional path to save the plot
    """
    from tensorboard.backend.event_processing import event_accumulator
    
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if len(event_files) == 0:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    
    # Load events
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot train loss
    if 'epoch/train_loss' in tags['scalars']:
        train_loss = ea.Scalars('epoch/train_loss')
        epochs = [x.step for x in train_loss]
        values = [x.value for x in train_loss]
        axes[0, 0].plot(epochs, values, label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot validation loss
    if 'epoch/val_loss' in tags['scalars']:
        val_loss = ea.Scalars('epoch/val_loss')
        epochs = [x.step for x in val_loss]
        values = [x.value for x in val_loss]
        axes[0, 1].plot(epochs, values, label='Val Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot perplexity
    if 'epoch/train_perplexity' in tags['scalars']:
        train_ppl = ea.Scalars('epoch/train_perplexity')
        epochs = [x.step for x in train_ppl]
        values = [x.value for x in train_ppl]
        axes[1, 0].plot(epochs, values, label='Train Perplexity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].set_title('Training Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'train/learning_rate' in tags['scalars']:
        lr = ea.Scalars('train/learning_rate')
        steps = [x.step for x in lr]
        values = [x.value for x in lr]
        axes[1, 1].plot(steps, values, label='Learning Rate', color='green')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_structures(
    structure1: np.ndarray,
    structure2: np.ndarray,
    output_path: Optional[str] = None
) -> Dict:
    """
    Compare two structures and compute similarity metrics.
    
    Args:
        structure1: First 3D structure array
        structure2: Second 3D structure array
        output_path: Optional path to save comparison
        
    Returns:
        Dictionary with comparison metrics
    """
    # Ensure same shape
    assert structure1.shape == structure2.shape, "Structures must have same shape"
    
    # Compute metrics
    total_blocks = structure1.size
    matching_blocks = np.sum(structure1 == structure2)
    accuracy = matching_blocks / total_blocks
    
    # Block type distribution comparison
    unique1, counts1 = np.unique(structure1, return_counts=True)
    unique2, counts2 = np.unique(structure2, return_counts=True)
    
    dist1 = dict(zip(unique1, counts1 / total_blocks))
    dist2 = dict(zip(unique2, counts2 / total_blocks))
    
    # Compute KL divergence
    all_blocks = set(unique1) | set(unique2)
    kl_div = 0.0
    for block in all_blocks:
        p = dist1.get(block, 1e-10)
        q = dist2.get(block, 1e-10)
        kl_div += p * np.log(p / q)
    
    metrics = {
        'total_blocks': int(total_blocks),
        'matching_blocks': int(matching_blocks),
        'accuracy': float(accuracy),
        'kl_divergence': float(kl_div)
    }
    
    print("\n" + "="*50)
    print("Structure Comparison")
    print("="*50)
    print(f"Matching blocks: {matching_blocks}/{total_blocks} ({accuracy:.2%})")
    print(f"KL Divergence: {kl_div:.4f}")
    print("="*50)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    print("Utility functions loaded")
    print("\nAvailable functions:")
    print("  - visualize_structure_3d")
    print("  - visualize_structure_2d_slices")
    print("  - analyze_dataset_statistics")
    print("  - plot_training_curves")
    print("  - compare_structures")
