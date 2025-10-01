import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model.decoder import DecoderBlock
from torchvision.models import Swin_T_Weights, Swin_B_Weights, Swin_V2_B_Weights
import math
from ultralytics import YOLO
import numpy as np
from data.transformations import CenterTransform
import random

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
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

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

    def forward2(self, img_t, img_t_minus_1):
        # TODO
        # Extract Features
        memory_t, B, H, W, C = self.extract_features(img_t)
        memory_t_minus_1, _, _, _, _ = self.extract_features(img_t_minus_1)
        memory_diff = memory_t - memory_t_minus_1 # TODO: Always positive?

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


class YOLODetector(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = YOLO(f'{path}/best_model.pt')
        # YOLO needs the non-normalized img
        transform = CenterTransform(augment=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = torch.tensor(transform.normalize.mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(transform.normalize.std).view(-1, 1, 1).to(device)

    def forward(self, img):
        img = img * self.std + self.mean
        preds = self.model(img, verbose=False)[0]
        boxes = preds.boxes.xyxyn
        x_mid = (boxes[:, 0] + boxes[:, 2]) / 2
        y_mid = (boxes[:, 1] + boxes[:, 3]) / 2
        midpoints = torch.stack((x_mid, y_mid), axis=1)
        confs = preds.boxes.conf.unsqueeze(0).T
        result = torch.cat((midpoints, confs), 1)
        return result

class PerfectDetector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, centers):
        return centers

class RandomDetector(nn.Module):
    def __init__(self, max_preds):
        super().__init__()
        self.max_preds = max_preds

    def forward(self, img):
        nr_of_pts = random.randint(0, self.max_preds)
        result = torch.rand(nr_of_pts, 3) 
        # If nr_of_pts = 1, then dimension is not correct: [] vs [[]]
        # if nr_of_pts == 1:
         #    result = result.unsqueeze(0)
        return result


def initialize_model_from_config(config):
    """ Initialize CenterPredictor from config. """
    if config['backbone'] in ['swin_b', 'swin_t']:
        print('using swin...')
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
        print('using dinov3...')
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
    elif config['backbone'] == 'yolo':
        print('using yolo...')
        model = YOLODetector(
            path=f'experiments/{config["model_name"]}'
        )
    elif config['backbone'] == 'perfect':
        print('using perfect detector...')
        model = PerfectDetector()
    elif config['backbone'] == 'random':
        print('using random detector...')
        model = RandomDetector(max_preds=config["max_preds"])
    else:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")


    return model
