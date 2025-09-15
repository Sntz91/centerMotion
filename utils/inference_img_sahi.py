import torch
import random
import numpy as np
from model import CenterPredictor
from dataset import get_val_dataset
import matplotlib.pyplot as plt

DEVICE = 'cpu'
model = CenterPredictor()
model.load_state_dict(torch.load('experiments/Baseline/best_model.pt', map_location=DEVICE))
model.eval()

def predict(model, image, conf_threshold=0.5):
    """Get predictions for a single patch image (C,H,W)"""
    with torch.no_grad():
        preds = model(image.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
    
    # Apply sigmoid to confidence
    preds[:, 2] = torch.sigmoid(preds[:, 2])
    
    # Filter by confidence
    mask = preds[:, 2] > conf_threshold
    filtered = preds[mask]
    
    points = filtered[:, :2].numpy()  # normalized coords [0,1]
    scores = filtered[:, 2].numpy()
    
    return points, scores

def slice_image(image, slice_size=224, overlap=0.2):
    """Split (C,H,W) image into overlapping patches"""
    C, H, W = image.shape
    stride = int(slice_size * (1 - overlap))
    slices = []
    positions = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = y, x
            y2, x2 = min(y + slice_size, H), min(x + slice_size, W)
            
            patch = torch.zeros((C, slice_size, slice_size), dtype=image.dtype)
            patch[:, :y2 - y1, :x2 - x1] = image[:, y1:y2, x1:x2]
            slices.append(patch)
            positions.append((x1, y1))
    
    return slices, positions

def predict_sliced(model, image, conf_threshold=0.5, slice_size=224, overlap=0.2):
    slices, positions = slice_image(image, slice_size, overlap)
    all_points = []
    all_scores = []

    for patch, (x_off, y_off) in zip(slices, positions):
        points, scores = predict(model, patch, conf_threshold)
        # scale normalized coords [0,1] to patch size
        points[:, 0] = points[:, 0] * patch.shape[2]  # width
        points[:, 1] = points[:, 1] * patch.shape[1]  # height
        # shift to full image coordinates
        points[:, 0] += x_off
        points[:, 1] += y_off
        all_points.append(points)
        all_scores.append(scores)
    
    if all_points:
        all_points = np.vstack(all_points)
        all_scores = np.hstack(all_scores)
    else:
        all_points = np.zeros((0, 2))
        all_scores = np.zeros(0)
    
    return all_points, all_scores

# --- Main evaluation ---
val_dataset = get_val_dataset()
idx = random.randint(0, len(val_dataset)-1)
item = val_dataset[idx]

pred_points, pred_scores = predict_sliced(model, item["img_t"], conf_threshold=0.5)
plt.imshow(item["img_t"].permute(1, 2, 0))
plt.scatter(pred_points[:, 0], pred_points[:, 1], marker='x', color='red', s=50)
plt.show()
