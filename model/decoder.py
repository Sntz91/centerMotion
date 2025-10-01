import torch
import torch.nn as nn
from model.rope import apply_2d_rotary_pos_emb

# decoder
class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, attention_dropout, dropout_1, dropout_2, dropout_3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=attention_dropout)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=attention_dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_1)
        self.dropout2 = nn.Dropout(p=dropout_2)
        self.dropout3 = nn.Dropout(p=dropout_3)

    def forward(self, q, m, H, W):
        # 1: Self attention on queries
        q_norm = self.norm1(q)
        attn_out = self.self_attn(q_norm, q_norm, q_norm)[0] # 0 for outputs only, 1 would be weights (viz)
        q = q + self.dropout1(attn_out) # Residual Connection + Dropout

        # 2: Cross attention: queries attend to memory
        q_norm = self.norm2(q)
        m_rot, m_rot_k = apply_2d_rotary_pos_emb(m, m, H, W) # apply ROPE
        attn_out = self.cross_attn(q_norm, m_rot, m_rot_k)[0]
        q = q + self.dropout2(attn_out)

        # 3: Feed Forward
        q_norm = self.norm3(q)
        ffn_out = self.ff(q_norm)
        q = q + self.dropout3(ffn_out)

        return q
