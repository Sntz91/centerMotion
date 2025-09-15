import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model.rope import apply_2d_rotary_pos_emb
from torchvision.models import Swin_T_Weights, Swin_B_Weights, Swin_V2_B_Weights
import math

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

class CenterPredictor(nn.Module):
    def __init__(self, backbone='swin_b', backbone_output_dim=None, hidden_dim=1024, 
                 patch_size=None, num_decoders=6, max_preds=50, n_attention_heads=4,
                 attention_dropout=0.1, dropout_1=0.1, dropout_2=0.1, dropout_3=0.1,
                 dinov3_repo_path=None, dinov3_ckpt_path=None):
        super().__init__()
        # Choose backbone
        self.backbone_type = backbone

        if backbone == 'swin_t':
            swin_model = models.swin_t(weights=Swin_T_Weights.DEFAULT)
            self.backbone = nn.Sequential(swin_model.features, swin_model.norm)
            backbone_output_dim = backbone_output_dim or 768
            patch_size = patch_size or 7

        elif backbone == 'swin_b':
            swin_model = models.swin_b(weights=Swin_B_Weights.DEFAULT)
            self.backbone = nn.Sequential(swin_model.features, swin_model.norm)
            backbone_output_dim = backbone_output_dim or 1024
            patch_size = patch_size or 7

        elif backbone == 'swin_v2_b':
            swin_model = models.swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            self.backbone = nn.Sequential(swin_model.features, swin_model.norm)
            backbone_output_dim = backbone_output_dim or 1024
            patch_size = patch_size or 7

        elif backbone == 'dinov3_vitl16': 
            assert dinov3_repo_path is not None, "Dino v3 repo path required."
            assert dinov3_ckpt_path is not None, "Dino v3 checkpoint path required."

            self.backbone = torch.hub.load(dinov3_repo_path, 'dinov3_vitl16', source='local', pretrained=False)
            # Load checkpoint: Online not available currently
            state_dict = torch.load(dinov3_ckpt_path, map_location='cpu')
            # state_dict = ckpt["model"] #if "model" in ckpt else ckpt # dunno about this
            self.backbone.load_state_dict(state_dict, strict=False)

            backbone_output_dim = backbone_output_dim or self.backbone.embed_dim
            patch_size = patch_size or 16

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone_output_dim = backbone_output_dim
        self.patch_size = patch_size

        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(backbone_output_dim, n_attention_heads, attention_dropout, dropout_1, dropout_2, dropout_3)
            for _ in range(num_decoders)
        ])

        # Prediction Head
        self.output_head = nn.Sequential(
            nn.Linear(backbone_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # [x, y, objectness]
        )

        # Learnable queries
        self.query_embed = nn.Parameter(torch.randn(max_preds, backbone_output_dim))

    def extract_features(self, img):
        """ Extract features based on backbone type. """
        if self.backbone_type == 'dinov3_vitl16':
            feats = self.backbone.get_intermediate_layers(img, n=1, reshape=True)[0] 
            B, C, H, W = feats.shape
            memory = feats.permute(0, 2, 3, 1).reshape(B, H*W, C)
        else:
            feats = self.backbone(img)
            B, H, W, C = feats.shape
            memory = swin_feature_map.flatten(1, 2) # [B, H*W, C]
        return memory, B, H, W, C

    def forward(self, img):
        # Extract Features
        memory, B, H, W, C = self.extract_features(img)

        # Initialize Queries
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        # Pass through decoder
        out = queries
        for layer in self.decoder:
            out = layer(out, memory, H, W)

        # Generate predictions
        preds = self.output_head(out)
        xy = preds[..., :2].sigmoid() # xy coordinates in [0, 1]
        objectness = preds[..., 2:] # raw logits for BCEWithLogitsLoss
        preds = torch.cat([xy, objectness], dim=-1)

        return preds


def initialize_model_from_config(config):
    """ Initialize CenterPredictor from config. """
    if config['backbone'] in ['swin_b', 'swin_t']:
        model = CenterPredictor(
            backbone_output_dim=config["backbone_output_dim"],
            hidden_dim=config["hidden_dim"],
            patch_size=config["patch_size"],
            num_decoders=config["num_decoders"],
            max_preds=config["max_preds"],
            backbone=config["backbone"],
            n_attention_heads=config["n_attention_heads"],
            attention_dropout=config["attention_dropout"],
            dropout_1=config["dropout_1"],
            dropout_2=config["dropout_2"],
            dropout_3=config["dropout_3"],
        )
    elif config['backbone'] in ['dinov3_vitl16']:
        model = CenterPredictor(
            dinov3_repo_path='/home/tobias/projects/dinov3',
            dinov3_ckpt_path='/home/tobias/projects/dinov3/dinov3_vitl16.pth',
            backbone_output_dim=config["backbone_output_dim"],
            hidden_dim=config["hidden_dim"],
            patch_size=config["patch_size"],
            num_decoders=config["num_decoders"],
            max_preds=config["max_preds"],
            backbone=config["backbone"],
            n_attention_heads=config["n_attention_heads"],
            attention_dropout=config["attention_dropout"],
            dropout_1=config["dropout_1"],
            dropout_2=config["dropout_2"],
            dropout_3=config["dropout_3"],
        )
    else:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")


    return model
