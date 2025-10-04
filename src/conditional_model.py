"""
Conditional MinecraftGPT Model with Text Conditioning
GPT model that generates structures based on text descriptions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TextEncoder(nn.Module):
    """Encodes text descriptions into embeddings."""
    
    def __init__(
        self,
        text_vocab_size: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_text_length: int = 128
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Text embedding
        self.token_embedding = nn.Embedding(text_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_text_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text descriptions.
        
        Args:
            text_ids: Text token IDs (batch_size, text_len)
            text_mask: Padding mask (batch_size, text_len)
            
        Returns:
            Text embeddings (batch_size, text_len, d_model)
        """
        # Embed tokens
        x = self.token_embedding(text_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create attention mask (invert for PyTorch convention)
        if text_mask is not None:
            # text_mask: 1 = valid, 0 = padding
            # attention_mask: True = ignore, False = attend
            attention_mask = (text_mask == 0)
        else:
            attention_mask = None
        
        # Encode
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=attention_mask
        )
        
        return encoded


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class CrossAttention(nn.Module):
    """Cross-attention between structure and text."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,  # Structure embeddings
        key_value: torch.Tensor,  # Text embeddings
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention from structure to text.
        
        Args:
            query: Structure embeddings (batch_size, struct_len, d_model)
            key_value: Text embeddings (batch_size, text_len, d_model)
            mask: Text padding mask (batch_size, text_len)
            
        Returns:
            Attended structure embeddings (batch_size, struct_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key_value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(key_value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask (for padded text tokens)
        if mask is not None:
            # mask: (batch_size, text_len) with 1 = valid, 0 = padding
            # Expand for heads and query length
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, text_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Output projection
        output = self.out_linear(context)
        
        return output


class ConditionalTransformerBlock(nn.Module):
    """Transformer block with cross-attention to text."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention (causal)
        self.self_attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention to text
        self.cross_attention = CrossAttention(d_model, n_heads, dropout)
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with self-attention and cross-attention.
        
        Args:
            x: Structure embeddings (batch_size, struct_len, d_model)
            text_embeddings: Text embeddings (batch_size, text_len, d_model)
            causal_mask: Causal mask for self-attention
            text_mask: Padding mask for text
            
        Returns:
            Output embeddings (batch_size, struct_len, d_model)
        """
        # Self-attention with causal mask
        attn_output, _ = self.self_attention(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to text
        cross_output = self.cross_attention(x, text_embeddings, text_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class ConditionalMinecraftGPT(nn.Module):
    """
    Conditional GPT model for text-to-structure generation.
    Generates Minecraft structures based on text descriptions.
    """
    
    def __init__(
        self,
        block_vocab_size: int,
        text_vocab_size: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_seq_length: int = 4096,
        max_text_length: int = 128,
        dropout: float = 0.1,
        text_encoder_layers: int = 4
    ):
        super().__init__()
        
        self.block_vocab_size = block_vocab_size
        self.text_vocab_size = text_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.use_gradient_checkpointing = False  # Can be enabled to save memory
        
        # Text encoder
        self.text_encoder = TextEncoder(
            text_vocab_size=text_vocab_size,
            d_model=d_model,
            n_layers=text_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_text_length=max_text_length
        )
        
        # Block embedding
        self.block_embedding = nn.Embedding(block_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Conditional transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ConditionalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, block_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        block_ids: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            block_ids: Block token IDs (batch_size, seq_len)
            text_ids: Text token IDs (batch_size, text_len)
            text_mask: Text padding mask (batch_size, text_len), 1=valid, 0=pad
            
        Returns:
            Logits (batch_size, seq_len, block_vocab_size)
        """
        batch_size, seq_len = block_ids.shape
        device = block_ids.device
        
        # Encode text description
        text_embeddings = self.text_encoder(text_ids, text_mask)
        
        # Embed blocks
        x = self.block_embedding(block_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Pass through conditional transformer blocks
        # Use gradient checkpointing during training to save memory
        if self.training and hasattr(self, 'use_gradient_checkpointing') and self.use_gradient_checkpointing:
            for block in self.transformer_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, text_embeddings, causal_mask, text_mask, use_reentrant=False
                )
        else:
            for block in self.transformer_blocks:
                x = block(x, text_embeddings, causal_mask, text_mask)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        text_prompt: str,
        text_tokenizer,
        start_token_id: int,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate structure from text prompt.
        
        Args:
            text_prompt: Text description of desired structure
            text_tokenizer: TextTokenizer instance
            start_token_id: Start token ID for blocks
            max_new_tokens: Maximum number of blocks to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            device: Device to generate on
            
        Returns:
            Generated block sequence (1, seq_len)
        """
        self.eval()
        
        # Encode text prompt
        text_ids = text_tokenizer.encode(text_prompt, max_length=128)
        text_ids = torch.tensor([text_ids], dtype=torch.long).to(device)
        text_mask = (text_ids != 0).long()  # Assuming PAD=0
        
        # Encode text once (will be reused)
        text_embeddings = self.text_encoder(text_ids, text_mask)
        
        # Start with start token
        generated = torch.tensor([[start_token_id]], dtype=torch.long).to(device)
        
        for _ in range(max_new_tokens):
            # Get logits for next token
            block_embeddings = self.block_embedding(generated) * math.sqrt(self.d_model)
            block_embeddings = self.positional_encoding(block_embeddings)
            
            # Create causal mask
            seq_len = generated.size(1)
            causal_mask = self.create_causal_mask(seq_len, device)
            
            # Pass through transformer blocks
            x = block_embeddings
            for block in self.transformer_blocks:
                x = block(x, text_embeddings, causal_mask, text_mask)
            
            # Get logits for last position
            logits = self.output_projection(x[:, -1, :]) / temperature
            
            # Apply sampling strategies
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            if generated.size(1) >= self.max_seq_length:
                break
        
        return generated
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from vocab import VOCAB_SIZE as BLOCK_VOCAB_SIZE
    
    print("Testing ConditionalMinecraftGPT...")
    
    # Create model
    model = ConditionalMinecraftGPT(
        block_vocab_size=BLOCK_VOCAB_SIZE,
        text_vocab_size=2000,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        max_seq_length=4096,
        max_text_length=128
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 2
    block_seq_len = 100
    text_seq_len = 20
    
    block_ids = torch.randint(0, BLOCK_VOCAB_SIZE, (batch_size, block_seq_len))
    text_ids = torch.randint(5, 100, (batch_size, text_seq_len))  # Avoid special tokens
    text_mask = torch.ones(batch_size, text_seq_len)
    
    output = model(block_ids, text_ids, text_mask)
    
    print(f"\nInput shapes:")
    print(f"  Blocks: {block_ids.shape}")
    print(f"  Text: {text_ids.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✓ Model initialized and tested successfully!")
