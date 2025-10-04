"""
Text Tokenizer and Vocabulary for Description Prompts
Extracts and processes text descriptions from .schem filenames
"""

import re
import os
import glob
from typing import List, Dict, Set, Tuple
from collections import Counter
import json


# Special tokens for text
TEXT_SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<START>': 2,
    '<END>': 3,
    '<SEP>': 4,  # Separator between text and structure
}


class TextTokenizer:
    """Tokenizer for structure descriptions."""
    
    def __init__(self, vocab_size: int = 2000, min_freq: int = 2):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a word to be included
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        
        self.word_to_id = TEXT_SPECIAL_TOKENS.copy()
        self.id_to_word = {v: k for k, v in TEXT_SPECIAL_TOKENS.items()}
        
        self.current_id = len(TEXT_SPECIAL_TOKENS)
    
    def extract_description_from_filename(self, filename: str) -> str:
        """
        Extract description from .schem filename.
        
        Examples:
            "a_big_abandoned_barn_built_out_of_spruce_wood_with_interior_and_pointed_roof_0001.schem"
            -> "a big abandoned barn built out of spruce wood with interior and pointed roof"
            
        Args:
            filename: Filename with description
            
        Returns:
            Cleaned description text
        """
        # Remove path
        filename = os.path.basename(filename)
        
        # Remove extension
        filename = filename.replace('.schem', '')
        
        # Remove trailing numbers (e.g., _0001)
        filename = re.sub(r'_\d{4,}$', '', filename)
        
        # Replace underscores with spaces
        description = filename.replace('_', ' ')
        
        # Clean up multiple spaces
        description = re.sub(r'\s+', ' ', description)
        
        # Lowercase
        description = description.lower().strip()
        
        return description
    
    def build_vocab_from_files(self, file_paths: List[str]):
        """
        Build vocabulary from a list of .schem files.
        
        Args:
            file_paths: List of .schem file paths
        """
        print(f"Building vocabulary from {len(file_paths)} files...")
        
        # Extract all descriptions
        all_words = []
        
        for filepath in file_paths:
            description = self.extract_description_from_filename(filepath)
            words = self.tokenize_text(description)
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        print(f"Found {len(word_counts)} unique words")
        
        # Filter by frequency and limit vocab size
        filtered_words = [
            word for word, count in word_counts.most_common(self.vocab_size)
            if count >= self.min_freq
        ]
        
        print(f"Filtered to {len(filtered_words)} words (min_freq={self.min_freq})")
        
        # Build vocabulary
        for word in filtered_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.current_id
                self.id_to_word[self.current_id] = word
                self.current_id += 1
        
        print(f"Final vocabulary size: {len(self.word_to_id)}")
        
        # Print most common words
        print("\nTop 30 most common words:")
        for i, (word, count) in enumerate(word_counts.most_common(30), 1):
            print(f"  {i:2}. {word:20} ({count:4} occurrences)")
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Simple word-based tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of token IDs
        """
        words = self.tokenize_text(text)
        
        # Convert to IDs
        token_ids = [TEXT_SPECIAL_TOKENS['<START>']]
        
        for word in words[:max_length - 2]:  # Reserve space for <START> and <END>
            token_id = self.word_to_id.get(word, TEXT_SPECIAL_TOKENS['<UNK>'])
            token_ids.append(token_id)
        
        token_ids.append(TEXT_SPECIAL_TOKENS['<END>'])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        words = []
        
        for token_id in token_ids:
            if token_id in [TEXT_SPECIAL_TOKENS['<START>'], 
                           TEXT_SPECIAL_TOKENS['<END>'],
                           TEXT_SPECIAL_TOKENS['<PAD>']]:
                continue
            
            word = self.id_to_word.get(token_id, '<UNK>')
            if word != '<UNK>':
                words.append(word)
        
        return ' '.join(words)
    
    def pad_sequence(self, token_ids: List[int], max_length: int) -> List[int]:
        """
        Pad sequence to max_length.
        
        Args:
            token_ids: List of token IDs
            max_length: Target length
            
        Returns:
            Padded sequence
        """
        if len(token_ids) >= max_length:
            return token_ids[:max_length]
        
        padding = [TEXT_SPECIAL_TOKENS['<PAD>']] * (max_length - len(token_ids))
        return token_ids + padding
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file."""
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': len(self.word_to_id),
            'min_freq': self.min_freq
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = {k: int(v) for k, v in vocab_data['word_to_id'].items()}
        self.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        self.min_freq = vocab_data['min_freq']
        self.current_id = len(self.word_to_id)
        
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {len(self.word_to_id)}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a tokenizer from a saved vocabulary file.
        
        Args:
            filepath: Path to vocabulary JSON file
            
        Returns:
            TextTokenizer instance with loaded vocabulary
        """
        tokenizer = cls()
        tokenizer.load_vocab(filepath)
        return tokenizer


def analyze_descriptions(data_dir: str):
    """
    Analyze descriptions in a dataset directory.
    
    Args:
        data_dir: Directory containing .schem files
    """
    file_paths = glob.glob(os.path.join(data_dir, "**/*.schem"), recursive=True)
    
    if len(file_paths) == 0:
        print(f"No .schem files found in {data_dir}")
        return
    
    print(f"\nAnalyzing {len(file_paths)} files...")
    
    tokenizer = TextTokenizer()
    
    descriptions = []
    description_lengths = []
    
    for filepath in file_paths[:20]:  # Show first 20 examples
        description = tokenizer.extract_description_from_filename(filepath)
        descriptions.append(description)
        words = tokenizer.tokenize_text(description)
        description_lengths.append(len(words))
    
    print("\n" + "="*80)
    print("Sample Descriptions:")
    print("="*80)
    
    for i, desc in enumerate(descriptions[:10], 1):
        print(f"{i:2}. {desc}")
    
    print("\n" + "="*80)
    print("Description Statistics:")
    print("="*80)
    
    import numpy as np
    print(f"Average words per description: {np.mean(description_lengths):.1f}")
    print(f"Min words: {np.min(description_lengths)}")
    print(f"Max words: {np.max(description_lengths)}")
    print(f"Median words: {np.median(description_lengths):.1f}")


def build_vocabulary_from_dataset(data_dir: str, output_path: str = "text_vocab.json"):
    """
    Build and save vocabulary from dataset.
    
    Args:
        data_dir: Directory containing .schem files
        output_path: Path to save vocabulary
    """
    file_paths = glob.glob(os.path.join(data_dir, "**/*.schem"), recursive=True)
    
    if len(file_paths) == 0:
        print(f"No .schem files found in {data_dir}")
        return None
    
    tokenizer = TextTokenizer(vocab_size=2000, min_freq=2)
    tokenizer.build_vocab_from_files(file_paths)
    tokenizer.save_vocab(output_path)
    
    return tokenizer


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Use the fixed_all_files directory
        data_dir = "fixed_all_files (1)/fixed_all_files"
    
    print(f"Analyzing descriptions in: {data_dir}")
    
    # Analyze descriptions
    analyze_descriptions(data_dir)
    
    # Build vocabulary
    print("\n" + "="*80)
    print("Building Vocabulary")
    print("="*80)
    
    tokenizer = build_vocabulary_from_dataset(data_dir, "text_vocab.json")
    
    if tokenizer:
        # Test encoding/decoding
        print("\n" + "="*80)
        print("Testing Encoding/Decoding")
        print("="*80)
        
        test_descriptions = [
            "a big medieval house with oak wood and stone",
            "small abandoned barn built out of spruce",
            "fantasy house with red roof and interior"
        ]
        
        for desc in test_descriptions:
            encoded = tokenizer.encode(desc)
            decoded = tokenizer.decode(encoded)
            print(f"\nOriginal:  {desc}")
            print(f"Encoded:   {encoded[:20]}..." if len(encoded) > 20 else f"Encoded:   {encoded}")
            print(f"Decoded:   {decoded}")
