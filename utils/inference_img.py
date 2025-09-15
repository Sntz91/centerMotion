import torch
import random
from model import CenterPredictor
from dataset import get_val_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

model = CenterPredictor()
model.load_state_dict(
    torch.load('experiments/Baseline/best_model.pt', map_location=torch.device('cpu'))
)
model.eval()
DEVICE = 'cpu'

def predict(model, image, conf_threshold):
    """Get predictions for a single image"""
    with torch.no_grad():
        preds = model(image.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
    
    # Apply sigmoid to confidence scores
    preds[:, 2] = torch.sigmoid(preds[:, 2])
    
    # Filter by confidence
    mask = preds[:, 2] > conf_threshold
    filtered_preds = preds[mask]
    
    points = filtered_preds[:, :2].numpy()
    scores = filtered_preds[:, 2].numpy()
    
    return points, scores

val_dataset = get_val_dataset()
idx = random.randint(0, len(val_dataset))
item = val_dataset[idx]

# for item in tqdm(val_dataset):
pred_points, pred_scores = predict(model, item["img_t"], conf_threshold=0.5)
plt.imshow(item["img_t"].permute(1, 2, 0))
pred_points = pred_points * 224
plt.scatter(pred_points[:, 0], pred_points[:, 1], marker='x', color='red', s=50)
plt.show()
    

