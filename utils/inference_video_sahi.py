import cv2
import torch
from PIL import Image
from model import CenterPredictor
from dataset import CenterTransform  # your transform class
import numpy as np

DEVICE = 'cpu'

# --- VIDEO & MODEL SETUP ---
video_fname = "/home/tobias/data/gta2tv/processed/scenario_1/synced_camera_1.mp4"
model = CenterPredictor()
model.load_state_dict(torch.load('experiments/Baseline/best_model.pt', map_location=DEVICE))
model.eval()

val_transform = CenterTransform(augment=False)  # preprocessing same as validation

# --- MODEL PREDICTION FOR ONE PATCH ---
def predict(model, image_tensor, conf_threshold=0.5):
    """Predict points for a single 224x224 patch"""
    with torch.no_grad():
        preds = model(image_tensor.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()

    preds[:, 2] = torch.sigmoid(preds[:, 2])
    mask = preds[:, 2] > conf_threshold
    filtered = preds[mask]

    points = filtered[:, :2].numpy()  # normalized coords [0,1]
    scores = filtered[:, 2].numpy()
    return points, scores

# --- VIDEO CAPTURE & SAHI-STYLE INFERENCE ---
cap = cv2.VideoCapture(video_fname)
cv2.namedWindow("video", cv2.WINDOW_NORMAL)

slice_size = 224
overlap = 0.2
skip = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    skip += 1
    if skip <= 100:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    stride = int(slice_size * (1 - overlap))
    slices = []
    positions = []

    # Slice full-resolution frame
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + slice_size, H)
            x2 = min(x + slice_size, W)
            patch = frame_rgb[y:y2, x:x2]
            slices.append(patch)
            positions.append((x, y))

    all_points = []

    for patch, (x_off, y_off) in zip(slices, positions):
        patch_pil = Image.fromarray(patch)
        patch_tensor, _ = val_transform(patch_pil, torch.empty((0, 2)))  # dummy centers

        # Predict on patch
        points, _ = predict(model, patch_tensor, conf_threshold=0.5)

        # Scale normalized points to patch pixels
        points[:, 0] *= patch_tensor.shape[2]  # width
        points[:, 1] *= patch_tensor.shape[1]  # height

        # Shift to original frame coordinates
        points[:, 0] += x_off
        points[:, 1] += y_off

        all_points.append(points)

    if all_points:
        all_points = np.vstack(all_points)
    else:
        all_points = np.zeros((0, 2))

    # Draw points on frame
    for x, y in all_points:
        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), 5)

    cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
