"""MMDG-inspired bidirectional cross-attention adapted to MCSAT token streams."""
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, query, context):
        attended, _ = self.attention(self.norm_q(query), self.norm_kv(context), self.norm_kv(context), need_weights=False)
        return query + self.output(attended)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.acoustic_from_vibration = CrossAttention(dim, heads, dropout)
        self.vibration_from_acoustic = CrossAttention(dim, heads, dropout)

    def forward(self, acoustic_tokens, vibration_tokens):
        acoustic = self.acoustic_from_vibration(acoustic_tokens, vibration_tokens)
        vibration = self.vibration_from_acoustic(vibration_tokens, acoustic_tokens)
        return __import__('torch').cat((acoustic, vibration), dim=1)
