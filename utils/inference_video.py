# Load video in cv2
# scale image to get predictions
# Filter by conf score
# upscale boxes for video


import cv2
import torch
from PIL import Image
from model import CenterPredictor
import torchvision.transforms.v2 as transforms
from dataset import CenterTransform  
import yaml

# --- VIDEO PATH & MODEL ---
video_fname = "/home/tobias/data/gta2tv/processed/scenario_1/synced_camera_1.mp4"
with open("config.yaml") as f:
    config = yaml.safe_load(f)
model = CenterPredictor(
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
model.load_state_dict(
    torch.load('experiments/Baseline_w_more_preds/best_model.pt', map_location=torch.device('cpu'))
)
model.eval()

# --- TRANSFORM (same as validation pipeline) ---
val_transform = CenterTransform(augment=False)

# --- VIDEO CAPTURE ---
cap = cv2.VideoCapture(video_fname)
cv2.namedWindow("video", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) → RGB (PIL expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # Apply SAME preprocessing as dataloader
    # (dummy empty centers, since transform expects centers too)
    image, _ = val_transform(frame_pil, torch.empty((0, 2)))

    H, W, _ = frame.shape

    with torch.no_grad():
        preds = model(image.unsqueeze(0)).squeeze(0)

    # Objectness sigmoid + threshold
    preds[:, 2] = torch.sigmoid(preds[:, 2])
    mask = preds[:, 2] > 0.5
    preds = preds[mask][:, :2]

    # Scale back to original frame size
    preds[:, 0] = preds[:, 0] * W
    preds[:, 1] = preds[:, 1] * H

    # Draw points
    for x, y in preds:
        cv2.circle(frame, (int(x.item()), int(y.item())), 5, (255, 0, 0), 5)

    # Show updated frame
    cv2.imshow('video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
