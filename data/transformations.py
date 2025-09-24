import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import random
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from multimethod import multimethod

@dataclass
class AugmentationConfig:
    """ Configuration for data augmentation parameters. """
    # Color augmentations
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.3
    color_jitter_hue: float = 0.05

    # Crop augmentations
    crop_scale_range: List[float] = None
    crop_ratio_range: List[float] = None
    crop_prob: float = 0.5

    # Flip augmentations
    flip_prob: float = 0.5

    # Affine augmentations
    rotation_angle_range: List[float] = None
    scale_range: List[float] = None
    translation_range: List[float] = None

    # BLur augmentations
    gaussian_blur_prob: float = 0.3
    gaussian_blur_kernel_size: List[int] = None

    def __post_init__(self):
        # Set default values for list parameters
        if self.crop_scale_range is None:
            self.crop_scale_range = [0.5, 1.0]
        if self.crop_ratio_range is None:
            self.crop_ratio_range = [0.75, 1.33]
        if self.rotation_angle_range is None:
            self.rotation_angle_range = [-15, 15]
        if self.scale_range is None:
            self.scale_range = [0.9, 1.1]
        if self.translation_range is None:
            self.translation_range = [-0.1, 0.1]
        if self.gaussian_blur_kernel_size is None:
            self.gaussian_blur_kernel_size = [3, 5]
    


class CenterTransform:
    """ Handles image- & coordinate transformations. """
    def __init__(self, img_size: Tuple[int, int] = (224, 224), augment: bool = True, 
                 aug_config: Optional[AugmentationConfig] = None):
        self.img_size = img_size
        self.augment = augment
        self.aug_config = aug_config or AugmentationConfig()

        # Standard normalizatino for ImageNet pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Color jitter transformations
        self.color_jitter = transforms.ColorJitter(
            brightness=self.aug_config.color_jitter_brightness,
            contrast=self.aug_config.color_jitter_contrast,
            saturation=self.aug_config.color_jitter_saturation,
            hue=self.aug_config.color_jitter_hue
        )

    @multimethod
    def __call__(self, image: Image.Image, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Call function for using only one image """
        if self.augment:
            params = self._sample_augmentation_params(image.size)
            image, centers = self._apply_augmentations(image, centers, params)
        return self._finalize(image, centers)

    @multimethod
    def __call__(self, img_prev: Image.Image, img_curr: Image.Image, centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Call function for adding t-1 image """
        if self.augment:
            params = self._sample_augmentation_params(img_curr.size)
            img_prev, _ = self._apply_augmentations(img_prev, torch.empty(0), params)
            img_curr, centers = self._apply_augmentations(img_curr, centers, params)
        img_prev, _ = self._finalize(img_prev, torch.empty(0))
        img_curr, centers = self._finalize(img_curr, centers)
        return img_prev, img_curr, centers

    def _finalize(self, image, centers):
        # Always resize to target size and normalize
        image = image.resize(self.img_size)
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        # Ensure centers is always 2D [N, 2]
        if centers.numel() > 0 and centers.dim() == 1:
            centers = centers.unsqueeze(0)

        return image, centers

    def _sample_augmentation_params(self, img_size: Tuple[int, int]) -> Dict[str, Any]:
        width, height = img_size

        params = {
            "crop_params": None,
            "do_hflip": torch.rand(1) < self.aug_config.flip_prob,
            "do_vflip": torch.rand(1) < self.aug_config.flip_prob,
            "angle": random.uniform(*self.aug_config.rotation_angle_range),
            "scale": random.uniform(*self.aug_config.scale_range),
            "tx": random.uniform(*self.aug_config.translation_range) * width,
            "ty": random.uniform(*self.aug_config.translation_range) * height,
            "do_color": True,
            "do_blur": torch.rand(1) < self.aug_config.gaussian_blur_prob,
            "blur_kernel": random.choice(self.aug_config.gaussian_blur_kernel_size),
        }

        if torch.rand(1) < self.aug_config.crop_prob:
            params["crop_params"] = self._get_random_crop_params(width, height)

        if params["do_color"]:
            fn_idx, brightness, contrast, saturation, hue = transforms.ColorJitter.get_params(
                brightness=[max(0, 1 - self.aug_config.color_jitter_brightness), 1 + self.aug_config.color_jitter_brightness],
                contrast=[max(0, 1 - self.aug_config.color_jitter_contrast), 1 + self.aug_config.color_jitter_contrast],
                saturation=[max(0, 1 - self.aug_config.color_jitter_saturation), 1 + self.aug_config.color_jitter_saturation],
                hue=[-self.aug_config.color_jitter_hue, self.aug_config.color_jitter_hue]
            )
        params["color_jitter"] = (fn_idx, brightness, contrast, saturation, hue)

        return params

    def _apply_augmentations(self, image: Image.Image, centers: torch.Tensor, params: Dict[str, Any]) -> Tuple[Image.Image, torch.Tensor]:
        original_size = image.size[::-1] 

        # 1. Random Crop (changes coordinate system -> Must be first)
        # if torch.rand(1) < self.aug_config.crop_prob:
        if params["crop_params"] is not None:
            image, centers = self._apply_random_crop(image, centers, original_size, params["crop_params"])

        # 2. Horizontal and vertical flips
        if params["do_hflip"]:
            image = TF.hflip(image)
            if centers.numel() > 0:
                centers[:, 0] = 1.0 - centers[:, 0]

        if params["do_vflip"]:
            image = TF.vflip(image)
            if centers.numel() > 0:
                centers[:, 1] = 1.0 - centers[:, 1]

        # 3. Affine transformations (rotation, scale, translation)
        image = TF.affine(
            image, 
            angle=params["angle"],
            translate=(int(params["tx"]), int(params["ty"])),
            scale=params["scale"],
            shear=0
        )
        if centers.numel() > 0:
            centers = self._transform_centers_for_affine(
                centers, image.size[0], image.size[1],
                params["angle"], (params["tx"], params["ty"]), params["scale"]
            )

        # 4. Color augmentations
        if params["do_color"]:
            fn_idx, brightness, contrast, saturation, hue = params["color_jitter"]
            for fn_id in fn_idx:
                if fn_id == 0 and brightness is not None:
                    image = TF.adjust_brightness(image, brightness)
                elif fn_id == 1 and contrast is not None:
                    image = TF.adjust_contrast(image, contrast)
                elif fn_id == 2 and saturation is not None:
                    image = TF.adjust_saturation(image, saturation)
                elif fn_id == 3 and hue is not None:
                    image = TF.adjust_hue(image, hue)

        # 5. Gaussian blur
        if params["do_blur"]:
            image = TF.gaussian_blur(image, kernel_size=params["blur_kernel"])

        return image, centers

    def _apply_random_crop(self, image: Image.Image, centers: torch.Tensor,
                           original_size: Tuple[int, int], crop_params: Tuple[int, int, int, int]) -> Tuple[Image.Image, torch.Tensor]:
        i, j, h, w = crop_params
        image = TF.crop(image, i, j, h, w)
        # No centers -> just crop
        if centers.numel() == 0:
            return image, centers
        centers = self._transform_centers_for_crop(centers, crop_params, original_size)
        return image, centers

    def _get_random_crop_params(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        area = img_height * img_width

        for _ in range(10): # Try up to 10 times
            target_area = random.uniform(*self.aug_config.crop_scale_range) * area
            log_ratio = (math.log(self.aug_config.crop_ratio_range[0]),
                         math.log(self.aug_config.crop_ratio_range[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= img_width and 0 < h <= img_height:
                i = random.randint(0, img_height - h)
                j = random.randint(0, img_width - w)
                return i, j, h, w

        # Fallback: center crop with valid aspect ratio
        in_ratio = img_width / img_height
        min_ratio, max_ratio = self.aug_config.crop_ratio_range

        if in_ratio < min_ratio:
            w = img_width
            h = int(round(w / min_ratio))
        elif in_ratio > max_ratio:
            h = img_height
            w = int(round(h * max_ratio))
        else:
            w, h = img_width, img_height

        i = (img_height - h) // 2
        j = (img_width - w) // 2
        return i, j, h, w

    def _transform_centers_for_crop(self, centers: torch.Tensor, crop_params: Tuple[int, int, int, int], 
                                   original_size: Tuple[int, int]) -> torch.Tensor:
         i, j, h, w = crop_params # top, left, height, width
         orig_h, orig_w = original_size

         # Convert to pixel coordinates
         cx_pixel = centers[:, 0] * orig_w
         cy_pixel = centers[:, 1] * orig_h

         # Apply crop offset
         cx_crop = cx_pixel - j
         cy_crop = cy_pixel - i

         # Convert back to normalized coordinates
         centers_new = centers.clone()
         centers_new[:, 0] = cx_crop / w
         centers_new[:, 1] = cy_crop / h

         # Filter out centers outside crop area
         valid_mask = ((centers_new[:, 0] >= 0) & (centers_new[:, 0] <= 1) & 
                     (centers_new[:, 1] >= 0) & (centers_new[:, 1] <= 1))

         return centers_new[valid_mask]

    def _transform_centers_for_affine(self, centers: torch.Tensor, width: int, height: int,
                                     angle: float, translate: Tuple[float, float], 
                                     scale: float) -> torch.Tensor:
        # Convert to pixel coordinates
        cx = centers[:, 0] * width
        cy = centers[:, 1] * height
        
        # Center around origin
        cx -= width / 2
        cy -= height / 2
        
        # Apply rotation
        rad = math.radians(angle)
        new_x = cx * math.cos(rad) - cy * math.sin(rad)
        new_y = cx * math.sin(rad) + cy * math.cos(rad)
        
        # Apply scaling
        new_x *= scale
        new_y *= scale
        
        # Translate back and apply translation
        new_x += width / 2 + translate[0]
        new_y += height / 2 + translate[1]
        
        # Convert back to normalized coordinates
        centers[:, 0] = new_x / width
        centers[:, 1] = new_y / height
        
        # Filter out centers outside image bounds
        valid_mask = ((centers[:, 0] >= 0) & (centers[:, 0] <= 1) & 
                     (centers[:, 1] >= 0) & (centers[:, 1] <= 1))
        
        return centers[valid_mask]

if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # --- File paths setup ---
    base_dir = '/home/tobias/projects/01-cv/08-centerMotion/inputs/train/'
    img_filename = 'frame_01046.jpg'
    centers_filename = 'frame_01046.txt'

    prev_img_filename = 'frame_01045.jpg'

    img_path = os.path.join(base_dir, img_filename)
    centers_path = os.path.join(base_dir, centers_filename)
    prev_img_path = os.path.join(base_dir, prev_img_filename)

    # --- Data loading ---
    try:
        # Load the image
        original_image = Image.open(img_path).convert('RGB')
        prev_image = Image.open(prev_img_path).convert('RGB')
        
        # Load the centers
        centers_array = np.loadtxt(centers_path)
        if centers_array.ndim == 1:
            centers_array = centers_array.reshape(1, -1)
        original_centers = torch.from_numpy(centers_array).float()

    except FileNotFoundError:
        print(f"Error: One of the files was not found.")
        print(f"Please check if these paths exist:")
        print(f"Image: {img_path}")
        print(f"Centers: {centers_path}")
        exit()
        
    # --- Apply transformation ---
    # We will use the default image size and augmentation config
    transform = CenterTransform(augment=True)
    augmented_image_tensor, augmented_prev_image_tensor, augmented_centers = transform(original_image, prev_image, original_centers.clone())
    # augmented_image_tensor, augmented_centers = transform(original_image, original_centers.clone())
    
    # --- Visualization ---
    
    # Convert augmented image tensor back to PIL Image for plotting
    # Denormalize the image tensor
    # mean = torch.tensor(transform.normalize.mean).view(-1, 1, 1)
    # std = torch.tensor(transform.normalize.std).view(-1, 1, 1)
    
    # denormalized_image_tensor = augmented_image_tensor * std + mean
    # denormalized_prev_image_tensor = augmented_prev_image_tensor * std + mean
    
    # Permute dimensions for plotting (C, H, W) -> (H, W, C)
    # denormalized_image = denormalized_image_tensor.permute(1, 2, 0).numpy()
    # denormalized_prev_image = denormalized_prev_image_tensor.permute(1, 2, 0).numpy()

    denormalized_image = augmented_image_tensor.permute(1, 2, 0).numpy()
    denormalized_prev_image = augmented_prev_image_tensor.permute(1, 2, 0).numpy()
    
    # Create the plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    
    # Original Image and Centers
    axes[0].imshow(original_image)
    if original_centers.numel() > 0:
        # Scale normalized coordinates to pixel coordinates for plotting
        x_coords = original_centers[:, 0] * original_image.width
        y_coords = original_centers[:, 1] * original_image.height
        axes[0].scatter(x_coords, y_coords, color='red', marker='x', s=100, label='Original Centers')
    axes[0].set_title('Original Image and Centers')
    axes[0].legend()
    axes[0].axis('off')

    # Augmented Image and Centers
    axes[1].imshow(denormalized_image)
    if augmented_centers.numel() > 0:
        # Scale normalized coordinates to pixel coordinates for plotting on the augmented image
        x_coords = augmented_centers[:, 0] * transform.img_size[0]
        y_coords = augmented_centers[:, 1] * transform.img_size[1]
        axes[1].scatter(x_coords, y_coords, color='cyan', marker='o', s=100, label='Augmented Centers')
    axes[1].set_title('Augmented Image and Centers')
    axes[1].legend()
    axes[1].axis('off')

    axes[2].imshow(denormalized_prev_image)
    axes[2].set_title('Prev image')
    axes[2].axis('off')

    diff_img = denormalized_image - denormalized_prev_image

    axes[3].imshow(diff_img)
    axes[3].set_title('DIFF image')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    # TODO PLOT DIFFERENCE OF IMAGES
