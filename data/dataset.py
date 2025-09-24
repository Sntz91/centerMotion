import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from data.transformations import CenterTransform, AugmentationConfig
from multimethod import multimethod


class CenterDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[CenterTransform]=None, include_boxes: bool=False, use_prev_img: bool=False):
        self.data_dir = data_dir
        self.transform = transform
        self.include_boxes = include_boxes
        self.use_prev_img = use_prev_img

        # Get all image filenames
        self.frame_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

        if len(self.frame_files) == 0:
            raise ValueError(f'No image files found in {data_dir}.')

    def __len__(self) -> int:
        return len(self.frame_files) - 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load image
        img_path = os.path.join(self.data_dir, self.frame_files[idx + 1])
        image = Image.open(img_path).convert("RGB")

        # Load centers
        centers = self._load_centers(idx + 1)

        result = {}

        if self.use_prev_img:
            img_prev_path = os.path.join(self.data_dir, self.frame_files[idx])
            image_prev = Image.open(img_prev_path).convert("RGB")
            if self.transform:
                image_prev, image, centers = self.transform(image_prev, image, centers) # TODO: change it that we dont need to put centers in
            result['img_t_minus_1'] = image_prev
        else:
            # Apply transformations
            if self.transform:
                image, centers = self.transform(image, centers)

        # Create ground truth with objectness
        gt = self._create_ground_truth(centers)

        result['img_t'] = image
        result['gt'] = gt
        result['img_id'] = torch.tensor(idx, dtype=torch.long)

        if self.include_boxes:
            boxes = self._load_boxes(idx + 1)
            result['boxes_t'] = boxes

        return result

    def _load_centers(self, frame_idx: int) -> torch.Tensor:
        """ Load center coordinates from txt file. """
        frame_name = os.path.splitext(self.frame_files[frame_idx])[0]
        txt_path = os.path.join(self.data_dir, f'{frame_name}.txt')

        if not os.path.exists(txt_path):
            return torch.empty(0, 2, dtype=torch.float32)

        with open(txt_path, 'r') as f:
            centers = [[float(x) for x in line.strip().split()] for line in f]

        return torch.tensor(centers, dtype=torch.float32) if centers else torch.empty(0, 4, dtype=torch.float32)

    def _load_boxes(self, frame_idx: int) -> torch.Tensor:
        """ Load bboxes from txt file. """
        frame_name = os.path.splitext(self.frame_files[frame_idx])[0]
        txt_path = os.path.join(self.data_dir, f'{frame_name}_boxes.txt')

        if not os.path.exists(txt_path):
            return torch.empty(0, 4, dtype=torch.float32)

        with open(txt_path, 'r') as f:
            boxes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    xtl, ytl, xbr, ybr = parts
                    boxes.append([float(xtl), float(ytl), float(xbr), float(ybr)])

        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 4, dtype=torch.float32)

    def _create_ground_truth(self, centers: torch.Tensor) -> torch.Tensor:
        """ Create ground truth tensor with objectness scores. """
        if centers.numel() == 0:
            return torch.empty(0, 3, dtype=torch.float32)

        # Ensure centers is 2D
        if centers.dim() == 1:
            centers = centers.unsqueeze(0)

        # Add objectness (all 1s for valid centers)
        num_centers = centers.shape[0]
        objectness = torch.ones(num_centers, 1, dtype=torch.float32)

        # Combine to [x, y, objectness]
        gt = torch.cat([centers, objectness], dim=1)
        return gt

    
def custom_collate_fn(batch: List[Dict[str, Any]], max_objects: int=50) -> Dict[str, torch.Tensor]:
    """ Custom collate function handling the variable number of objects. """

    # Stack images
    images = torch.stack([item['img_t'] for item in batch])
    img_ids = torch.stack([item['img_id'] for item in batch])

    # Handle ground truth with padding
    gt_list = [item['gt'] for item in batch]
    padded_gt = [] # Resulting vector with gts & paddings included
    lengths = [] # How long is the part of the vector with the real values

    padding_vector = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

    for gt in gt_list:
        if gt.numel() == 0: # No objects
            gt = torch.empty(0, 3, dtype=torch.float32)
        elif gt.dim() == 1: # Single object
            gt = gt.unsqueeze(0)

        num_objects = gt.shape[0]
        lengths.append(num_objects)

        # Pad to max_objects
        if num_objects < max_objects:
            padding_length = max_objects - num_objects
            padding = padding_vector.unsqueeze(0).repeat(padding_length, 1)
            gt_padded = torch.cat([gt, padding], dim=0)
        # Truncate if too many
        else:
            gt_padded = gt[:max_objects] 
        padded_gt.append(gt_padded)

    # Stack all ground truths
    padded_gt = torch.stack(padded_gt)
    lengths = torch.tensor(lengths, dtype=torch.long)

    result = {
        'img_id': img_ids,
        'img_t': images,
        'gt_t': padded_gt,
        'lengths_t': lengths
    }

    if 'boxes_t' in batch[0]:
        boxes = [item['boxes_t'] for item in batch]
        result['boxes_t'] = boxes

    if 'img_t_minus_1' in batch[0]:
        prev_images = torch.stack([item['img_t_minus_1'] for item in batch])
        result['img_t_minus_1'] = prev_images

    return result

def create_dataloaders(train_dir: str='inputs/train',
                       val_dir: str='inputs/val',
                       batch_size: int=8,
                       num_workers: int=4,
                       img_size: int=224,
                       include_boxes: bool=False,
                       use_prev_img: bool=False,
                       aug_config: Optional[AugmentationConfig]=None) -> Tuple[DataLoader, DataLoader]:
    # Create transforms
    train_transform = CenterTransform(
        img_size=(img_size, img_size),
        augment=True,
        aug_config=aug_config or AugmentationConfig()
    )

    val_transform = CenterTransform(
        img_size=(img_size, img_size),
        augment=False
    )

    # Create datasets
    train_dataset = CenterDataset(train_dir, transform=train_transform, include_boxes=include_boxes, use_prev_img=use_prev_img)
    val_dataset = CenterDataset(val_dir, transform=val_transform, include_boxes=include_boxes, use_prev_img=use_prev_img)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader


def create_val_dataloader_only(val_dir: str='inputs/val',
                               batch_size: int=8,
                               num_workers: int=4,
                               img_size: int=224,
                               include_boxes: bool=True) -> DataLoader:
    """ Create validation dataloader only for convenience. """
    val_transform = CenterTransform(img_size=(img_size, img_size), augment=False)
    val_dataset = CenterDataset(val_dir, transform=val_transform, include_boxes=include_boxes)

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def create_val_dataset_only(val_dir: str='inputs/val',
                            img_size: int=224,
                            include_boxes: bool=True) -> CenterDataset:
    """ Create validation dataset only for convenience. """
    val_transform = CenterTransform(img_size=(img_size, img_size), augment=False)
    val_dataset = CenterDataset(val_dir, transform=val_transform, include_boxes=include_boxes)
    return val_dataset

def prepare_dataset_from_config(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """ Create dataloaders from a config dict. """

    # Extract augmentation parameters
    aug_config = AugmentationConfig(
        color_jitter_brightness=config.get("color_jitter_brightness", 0.3),
        color_jitter_contrast=config.get("color_jitter_contrast", 0.3),
        color_jitter_saturation=config.get("color_jitter_saturation", 0.3),
        color_jitter_hue=config.get("color_jitter_hue", 0.05),
        crop_scale_range=config.get("crop_scale_range", [0.5, 1.0]),
        crop_ratio_range=config.get("crop_ratio_range", [0.75, 1.33]),
        crop_prob=config.get("crop_prob", 0.5),
        flip_prob=config.get("flip_prob", 0.5),
        rotation_angle_range=config.get("rotation_angle_range", [-15, 15]),
        scale_range=config.get("scale_range", [0.9, 1.1]),
        translation_range=config.get("translation_range", [-0.1, 0.1]),
        gaussian_blur_prob=config.get("gaussian_blur_prob", 0.3),
        gaussian_blur_kernel_size=config.get("gaussian_blur_kernel_size", [3, 5])
    )

    return create_dataloaders(
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
        img_size=config.get("img_size", 224),
        aug_config=aug_config,
        use_prev_img=config.get("use_prev_img", False)
    )

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import math
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_dataloader, val_dataloader = prepare_dataset_from_config(config)
    transform = CenterTransform(augment=True)
    # mean = torch.tensor(transform.normalize.mean).view(-1, 1, 1)
    # std = torch.tensor(transform.normalize.std).view(-1, 1, 1)

    for batch_idx, batch in enumerate(train_dataloader):
        img_t = batch['img_t']  # shape: [B, C, H, W]
        gt_t = batch['gt_t']    # ground truth trajectories
        lengths_t = batch['lengths_t']
        if config.get("use_prev_img", False):
            img_t_minus_1 = batch['img_t_minus_1']

        # img_t = img_t * std + mean

        B, C, H, W = img_t.size()

        ncols = 2
        nrows = math.ceil(B / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
        axes = axes.flatten()  # flatten to 1D list for easy indexing

        for i in range(B):
            img_np1 = img_t[i].permute(1, 2, 0).numpy()
            if config.get("use_prev_img", False):
                img_np2 = img_t_minus_1[i].permute(1, 2, 0).numpy()
                img_np = np.hstack((img_np1, img_np2))
                img_np = np.hstack((img_np, (img_np1 - img_np2)))
            else:
                img_np = img_np1
            xs = gt_t[i][:, 0] * W
            ys = gt_t[i][:, 1] * H

            axes[i].imshow(img_np)
            axes[i].scatter(xs, ys, c='red')
            axes[i].set_title(f"Sample {i}")
            axes[i].axis("off")

        # Hide any unused subplots
        for j in range(B, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

        break
