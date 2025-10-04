"""
Conditional Generation Script for Text-to-Structure Minecraft GPT
==================================================================

Generate Minecraft structures from text descriptions using the trained conditional model.

Usage:
    # Generate single structure
    python src/conditional_generate.py --prompt "a big medieval house with oak wood and stone base" --checkpoint checkpoints/conditional_model_best.pt
    
    # Generate multiple variations
    python src/conditional_generate.py --prompt "a small church with wooden roof" --checkpoint checkpoints/conditional_model_best.pt --num_samples 5
    
    # Batch generation from file
    python src/conditional_generate.py --prompts_file prompts.txt --checkpoint checkpoints/conditional_model_best.pt
    
    # Control generation parameters
    python src/conditional_generate.py --prompt "an abandoned castle" --temperature 0.9 --top_k 50 --top_p 0.95
"""

import torch
import torch.nn.functional as F
import yaml
import argparse
from pathlib import Path
import time
from tqdm import tqdm

from conditional_model import ConditionalMinecraftGPT
from text_tokenizer import TextTokenizer
from vocab import get_block_vocab
from schematic_parser import SchematicParser
import numpy as np


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config, block_vocab_size, text_vocab_size, device):
    """Load trained model from checkpoint."""
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    
    return model


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k: Keep only top k tokens with highest probability (top-k filtering).
        top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits


@torch.no_grad()
def generate_structure(model, text_prompt, text_tokenizer, block_vocab, config, device,
                      temperature=1.0, top_k=0, top_p=0.0, seed=None):
    """
    Generate a structure from a text prompt.
    
    Args:
        model: Trained conditional GPT model
        text_prompt: Text description (string)
        text_tokenizer: Text tokenizer
        block_vocab: Block vocabulary
        config: Configuration dict
        device: torch device
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        seed: Random seed for reproducibility
        
    Returns:
        blocks: Generated structure as 3D numpy array (16, 16, 16)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model.eval()
    
    # Encode text prompt
    text_ids = text_tokenizer.encode(text_prompt)
    max_text_length = config['conditional']['max_text_length']
    
    # Get PAD token ID
    pad_token_id = text_tokenizer.word_to_id.get('<PAD>', 0)
    
    # Pad or truncate
    if len(text_ids) < max_text_length:
        text_ids = text_ids + [pad_token_id] * (max_text_length - len(text_ids))
    else:
        text_ids = text_ids[:max_text_length]
    
    text_ids = torch.tensor([text_ids], dtype=torch.long).to(device)
    text_mask = (text_ids != pad_token_id).float()
    
    # Start with START token (BOS)
    start_token = block_vocab.get('<START>', block_vocab.get('<BOS>', 2))
    generated = torch.tensor([[start_token]], dtype=torch.long).to(device)
    
    # Get inverse vocabulary for decoding
    idx_to_block = {v: k for k, v in block_vocab.items()}
    
    # Generate sequence
    max_seq_length = 16 * 16 * 16  # 4096 blocks
    
    print(f"\nGenerating structure from prompt: '{text_prompt}'")
    print(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
    print("Progress:")
    
    with tqdm(total=max_seq_length, desc="Generating blocks") as pbar:
        for _ in range(max_seq_length):
            # Forward pass
            logits = model(generated, text_ids, text_mask)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # Sample from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Check for END token (EOS)
            end_token = block_vocab.get('<END>', block_vocab.get('<EOS>', 3))
            if next_token.item() == end_token:
                break
            
            pbar.update(1)
    
    # Convert to block sequence (remove START token)
    block_sequence = generated[0, 1:].cpu().numpy()
    
    # Remove END token if present
    end_token = block_vocab.get('<END>', block_vocab.get('<EOS>', 3))
    if len(block_sequence) > 0 and block_sequence[-1] == end_token:
        block_sequence = block_sequence[:-1]
    
    # Pad to full size if needed
    if len(block_sequence) < max_seq_length:
        pad_token = block_vocab['<PAD>']
        block_sequence = np.pad(
            block_sequence,
            (0, max_seq_length - len(block_sequence)),
            constant_values=pad_token
        )
    elif len(block_sequence) > max_seq_length:
        block_sequence = block_sequence[:max_seq_length]
    
    # Reshape to 3D (16, 16, 16) - using Y-Z-X order
    blocks = block_sequence.reshape(16, 16, 16)
    
    # Convert indices to block names
    block_names = np.empty((16, 16, 16), dtype=object)
    for y in range(16):
        for z in range(16):
            for x in range(16):
                idx = blocks[y, z, x]
                block_names[y, z, x] = idx_to_block.get(idx, 'air')
    
    print(f"\n✓ Generation complete! Generated {len(block_sequence)} blocks")
    
    return block_names


def save_structure(blocks, output_path, description="Generated structure"):
    """Save structure as .schem file."""
    parser = SchematicParser(target_size=(16, 16, 16))
    success = parser.array_to_schematic(blocks, output_path, description)
    if success:
        print(f"✓ Structure saved: {output_path}")
    else:
        print(f"✗ Failed to save structure: {output_path}")


def generate_variations(model, text_prompt, text_tokenizer, block_vocab, config, device,
                       num_samples=3, output_dir='generated', temperature=1.0, top_k=0, top_p=0.0):
    """Generate multiple variations of a structure from the same prompt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename from prompt
    safe_prompt = text_prompt.replace(' ', '_').replace('/', '_')[:50]
    
    for i in range(num_samples):
        print(f"\n{'='*80}")
        print(f"Generating variation {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Generate structure with different seed
        blocks = generate_structure(
            model, text_prompt, text_tokenizer, block_vocab, config, device,
            temperature=temperature, top_k=top_k, top_p=top_p, seed=i
        )
        
        generation_time = time.time() - start_time
        
        # Save structure
        output_path = output_dir / f"{safe_prompt}_var{i+1:02d}.schem"
        save_structure(blocks, output_path, description=text_prompt)
        
        print(f"Generation time: {generation_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description='Generate Minecraft structures from text prompts')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='File containing multiple prompts (one per line)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--text_vocab', type=str, default='text_vocab.json',
                        help='Path to text vocabulary file')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Directory to save generated structures')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of variations to generate per prompt')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random, default: 1.0)')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k sampling (0 = disabled, default: 0)')
    parser.add_argument('--top_p', type=float, default=0.0,
                        help='Top-p (nucleus) sampling (0 = disabled, default: 0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.prompt is None and args.prompts_file is None:
        parser.error("Either --prompt or --prompts_file must be provided")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    block_vocab = get_block_vocab()
    print(f"Block vocabulary size: {len(block_vocab)}")
    
    text_tokenizer = TextTokenizer.load(args.text_vocab)
    print(f"Text vocabulary size: {text_tokenizer.vocab_size}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(
        args.checkpoint, config,
        len(block_vocab), text_tokenizer.vocab_size,
        device
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate structures
    print("\n" + "="*80)
    print("Starting Generation")
    print("="*80)
    
    if args.prompt is not None:
        # Single prompt
        generate_variations(
            model, args.prompt, text_tokenizer, block_vocab, config, device,
            num_samples=args.num_samples,
            output_dir=output_dir,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
    
    elif args.prompts_file is not None:
        # Multiple prompts from file
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*80}")
            print(f"Prompt {i}/{len(prompts)}: {prompt}")
            print(f"{'='*80}")
            
            generate_variations(
                model, prompt, text_tokenizer, block_vocab, config, device,
                num_samples=args.num_samples,
                output_dir=output_dir,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
    
    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"All structures saved to: {output_dir}")


if __name__ == '__main__':
    main()
