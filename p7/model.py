import torch

#implementation of GPT architecture 
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size = 50000
    max_seq_len = 512
    # Model dimensions
    d_model = 512       # divisible by n_head
    n_head = 8
    d_ff = 4 * d_model # FFN dimension (4×d_model standard)
    num_layers = 8  # Number of layers
    dropout = 0.1# Regularization
    pad_token_id = 0   # Special tokens

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            dropout=config.dropout,
            batch_first=True  # (batch, seq, dim)
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )

    def forward(self, x, mask, key_padding_mask):
        h, _ = self.attn(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.positional_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_embed.weight  # weight tying

    def forward(self, input_ids, attention_mask=None):
        b, s = input_ids.shape
        device = input_ids.device

        pos = torch.arange(0, s, device=device).unsqueeze(0).expand(b, -1)
        x = self.token_embed(input_ids) + self.positional_embed(pos)
        x = self.dropout(x)

        # GPT causal mask: shape (s, s)
        mask = torch.triu(torch.full((s, s), float('-inf'), device=device), diagonal=1)

        # key padding mask: True = pad
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        # run through GPT blocks
        for block in self.blocks:
            x = block(x, mask, key_padding_mask)

        x = self.ln_f(x)
        return self.lm_head(x)
    
