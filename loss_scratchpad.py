import torch
import numpy as np
from data.dataset import create_val_dataset_only
import matplotlib.pyplot as plt


def get_max_heatmap_value(query_x, query_y, centers, widths, heights, k_factor):
    # This is the single-query function from the previous response, adapted for clarity.
    sigma_x_sq = (k_factor * widths)**2
    sigma_y_sq = (k_factor * heights)**2
    
    # Broadcast query coordinates against all M centers
    diff_x_sq = (query_x - centers[:, 0])**2
    diff_y_sq = (query_y - centers[:, 1])**2
    
    # Calculate anisotropic exponent (Shape M)
    exponent = - (diff_x_sq / (2 * sigma_x_sq) + diff_y_sq / (2 * sigma_y_sq))
    gaussians = torch.exp(exponent)
    max_intensity, _ = torch.max(gaussians, dim=0)
    return 1.0 - max_intensity # Scalar

K_FACTOR = 0.3

val_dataset = create_val_dataset_only(img_size=512)
item = val_dataset[1000]
pred_x = 382
pred_y = 32
box = item['boxes_t']
centers = box[:, :2] * 512
widths = box[:, 2] * 512
heights = box[:, 3] * 512
preds = torch.tensor([
    [425.2, 425],
    [382, 32.3]
])

print(get_max_heatmap_value(preds[1,0], preds[1,1], centers, widths, heights, K_FACTOR))
print(get_max_heatmap_value(centers[0, 0], centers[0, 1], preds, 50, 50, K_FACTOR))


# DRAWING
"""
img = item['img_t'].permute(1, 2, 0).numpy()
plt.imshow(img)


total_heatmap = torch.zeros((512, 512), dtype=torch.float32)
range_i = torch.arange(512, dtype=torch.float32)
X, Y = torch.meshgrid(range_i, range_i, indexing='xy') 
X_flat = X.flatten()
Y_flat = Y.flatten()
query_centers = centers
query_widths = widths
query_heights = heights

for idx in range(X_flat.shape[0]):
    x = X_flat[idx]
    y = Y_flat[idx]
        
    # Get the loss value (1.0 - max_intensity) at the current pixel (x, y)
    loss_value = get_max_heatmap_value(x, y, query_centers, query_widths, query_heights, K_FACTOR)
        
    # We need the intensity (max_intensity) for standard heatmap visualization (0=low, 1=high)
    intensity = loss_value
        
    # Map the intensity back to the (y, x) array index
    total_heatmap[int(y), int(x)] = intensity

plt.imshow(total_heatmap, cmap='jet', alpha=0.7)
plt.show()
"""


"""
val_dataset = create_val_dataset_only(img_size=512)
item = val_dataset[1000]

img = item['img_t'].permute(1, 2, 0).numpy()

plt.imshow(img)


range_i = torch.arange(512, dtype=torch.float32)
X, Y = torch.meshgrid(range_i, range_i, indexing='xy') 

total_heatmap = torch.zeros((512, 512), dtype=torch.float32)
num_points = centers.shape[0]
sigma_sq = 20.0**2

preds = torch.tensor([[425.3, 425], [10, 10]])
total_loss = 0

good_preds = torch.tensor([
    [425, 425]
])

preds = centers


for i in range(num_points):
    cx, cy = centers[i, 0], centers[i, 1]
    sigma_x, sigma_y = box[i, 2] * 512, box[i, 3] * 512
    diff_x_sq, diff_y_sq = (X - cx)**2, (Y - cy)**2

    sigma_x_sq = (sigma_x*0.3)**2
    sigma_y_sq = (sigma_y*0.3)**2

    exponent = - (diff_x_sq / (2 * sigma_x_sq) + diff_y_sq / (2 * sigma_y_sq))
    gaussian = torch.exp(exponent)
    total_heatmap = torch.maximum(total_heatmap, gaussian)
    inverted_heatmap = 1.0 - total_heatmap
   
print('---', inverted_heatmap[32, 382])

for pred in preds:
    print(int(pred[0]), int(pred[1]))
    loss = inverted_heatmap[int(pred[0]), int(pred[1])]
    total_loss += loss
    print('loss 1', loss)


plt.imshow(inverted_heatmap, alpha=0.7, cmap='jet')
plt.scatter(preds[:, 0], preds[:, 1])
plt.show()

# NOW: Penalize the model when there are objects where no point is 

total_heatmap = torch.zeros((512, 512), dtype=torch.float32)
num_points = centers.shape[0]
sigma_sq = 20.0**2

num_preds = preds.shape[0]

for i in range(num_preds):
    cx, cy = preds[i, 0], preds[i, 1]
    sigma_x, sigma_y = 50, 50
    diff_x_sq, diff_y_sq = (X - cx)**2, (Y - cy)**2

    sigma_x_sq = (sigma_x*0.3)**2
    sigma_y_sq = (sigma_y*0.3)**2

    exponent = - (diff_x_sq / (2 * sigma_x_sq) + diff_y_sq / (2 * sigma_y_sq))
    gaussian = torch.exp(exponent)
    total_heatmap = torch.maximum(total_heatmap, gaussian)
    inverted_heatmap = 1.0 - total_heatmap


plt.imshow(inverted_heatmap, alpha=0.7, cmap='jet')
plt.scatter(centers[:, 0], centers[:, 1])
plt.show()

for center in centers:
    loss = inverted_heatmap[int(center[0]), int(center[1])]
    total_loss += loss
    print('loss 2', loss)

print('loss: ', total_loss.item())
"""
