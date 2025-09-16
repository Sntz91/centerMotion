import torch
import yaml
import random
import numpy as np
from model.model import initialize_model_from_config
from data.dataset import create_val_dataset_only, CenterTransform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image


def load_model_and_config():
    """Load model and configuration"""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model_from_config(config).to(device)
    model.load_state_dict(
        torch.load('/home/tobias/experiments/BaselineDINO/best_model.pt', 
                  map_location=torch.device(device))
    )
    model.eval()
    
    return model, config, device


def predict(model, image, device, conf_threshold=0.5):
    """Get predictions for a single image"""
    with torch.no_grad():
        preds = model(image.unsqueeze(0).to(device)).squeeze(0).cpu()
    
    # Apply sigmoid to confidence scores
    preds[:, 2] = torch.sigmoid(preds[:, 2])
    
    # Filter by confidence
    mask = preds[:, 2] > conf_threshold
    filtered_preds = preds[mask]
    
    points = filtered_preds[:, :2].numpy()
    scores = filtered_preds[:, 2].numpy()
    
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
            
            # Create patch with padding if needed
            patch = torch.zeros((C, slice_size, slice_size), dtype=image.dtype)
            patch[:, :y2 - y1, :x2 - x1] = image[:, y1:y2, x1:x2]
            slices.append(patch)
            positions.append((x1, y1))
    
    return slices, positions


def predict_sliced(model, image, device, conf_threshold=0.5, slice_size=224, overlap=0.2):
    """Predict using SAHI-style slicing"""
    slices, positions = slice_image(image, slice_size, overlap)
    all_points = []
    all_scores = []
    
    for patch, (x_off, y_off) in zip(slices, positions):
        points, scores = predict(model, patch, device, conf_threshold)
        
        if len(points) > 0:
            # Scale normalized coords [0,1] to patch size
            points[:, 0] = points[:, 0] * patch.shape[2]  # width
            points[:, 1] = points[:, 1] * patch.shape[1]  # height
            
            # Shift to full image coordinates
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


def predict_on_image(model, config, device, conf_threshold=0.5, plot_boxes=False, use_sahi=False, 
                    slice_size=224, overlap=0.2):
    """Predict on a random image from validation dataset"""
    print(f"Running image prediction {'with SAHI' if use_sahi else 'without SAHI'}...")
    
    val_dataset = create_val_dataset_only()
    idx = random.randint(0, len(val_dataset) - 1)
    item = val_dataset[idx]
    
    if use_sahi:
        pred_points, pred_scores = predict_sliced(model, item["img_t"], device, 
                                                 conf_threshold, slice_size, overlap)
        # Points are already in image coordinates for SAHI
        pred_points_scaled = pred_points
    else:
        pred_points, pred_scores = predict(model, item["img_t"], device, conf_threshold)
        pred_points_scaled = pred_points * config['img_size']  # Scale to image size
    
    # Visualization
    fig, ax = plt.subplots(1)
    
    # Plot bounding boxes if requested
    if plot_boxes and 'boxes_t' in item:
        for i, rect_coords in enumerate(item['boxes_t']):
            x_min, y_min, x_max, y_max = rect_coords * config['img_size']
            width = x_max - x_min
            height = y_max - y_min
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=1, edgecolor='r', facecolor='none',
                label=f'Rectangle {i+1}'
            )
            ax.add_patch(rect)
    
    # Show image and predictions
    ax.imshow(item["img_t"].permute(1, 2, 0))
    ax.scatter(pred_points_scaled[:, 0], pred_points_scaled[:, 1], 
              marker='x', color='red', s=50)
    
    method = "SAHI" if use_sahi else "Standard"
    plt.title(f"{method} Image Prediction (Found {len(pred_points)} points)")
    plt.show()
    
    print(f"Found {len(pred_points)} predictions with confidence > {conf_threshold}")


def predict_on_video(model, device, video_path, conf_threshold=0.5, start_frame=100, 
                    use_sahi=False, slice_size=224, overlap=0.2):
    """Predict on video frames"""
    print(f"Running video prediction {'with SAHI' if use_sahi else 'without SAHI'}...")
    
    val_transform = CenterTransform(augment=False)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    frame_count = 0
    stride = int(slice_size * (1 - overlap)) if use_sahi else None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count < start_frame:
            continue
        
        H, W, _ = frame.shape
        
        if use_sahi:
            # SAHI processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_points = []
            
            # Process overlapping patches
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    y2 = min(y + slice_size, H)
                    x2 = min(x + slice_size, W)
                    
                    # Extract patch
                    patch_rgb = frame_rgb[y:y2, x:x2]
                    patch_pil = Image.fromarray(patch_rgb)
                    
                    # Apply preprocessing (dummy centers for transform)
                    patch_tensor, *_ = val_transform(patch_pil, torch.empty((0, 2)))
                    
                    # Predict on patch
                    points, scores = predict(model, patch_tensor, device, conf_threshold)
                    
                    if len(points) > 0:
                        # Scale normalized points to patch pixels
                        points[:, 0] *= patch_tensor.shape[2]  # width
                        points[:, 1] *= patch_tensor.shape[1]  # height
                        
                        # Shift to original frame coordinates
                        points[:, 0] += x
                        points[:, 1] += y
                        
                        all_points.append(points)
            
            # Combine all predictions
            if all_points:
                pred_points_scaled = np.vstack(all_points)
            else:
                pred_points_scaled = np.zeros((0, 2))
                
        else:
            # Standard processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Apply same preprocessing as dataloader
            image, *_ = val_transform(frame_pil, torch.empty((0, 2)))
            
            pred_points, pred_scores = predict(model, image, device, conf_threshold)
            
            # Scale predictions to frame dimensions
            pred_points_scaled = pred_points.copy()
            pred_points_scaled[:, 0] *= W
            pred_points_scaled[:, 1] *= H
        
        # Draw predictions on frame
        for x, y in pred_points_scaled:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
        
        # Show frame info
        method_text = "SAHI" if use_sahi else "Standard"
        cv2.putText(frame, f"Frame: {frame_count}, Points: {len(pred_points_scaled)}, Method: {method_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Video prediction completed")


def main():
    """Main function - choose what to run here"""
    
    # Load model once
    model, config, device = load_model_and_config()
    
    # Configuration
    video_path = "/home/tobias/data/gta2tv/processed/scenario_1/synced_camera_1.mp4"
    conf_threshold = 0.5
    
    # SAHI Configuration
    use_sahi = True  # Set to True to enable SAHI
    slice_size = 224
    overlap = 0.2
    
    # Run image prediction
    # predict_on_image(model, config, device, conf_threshold, plot_boxes=False, 
                    # use_sahi=use_sahi, slice_size=slice_size, overlap=overlap)
    
    # Run video prediction
    predict_on_video(model, device, video_path, conf_threshold, start_frame=100,
                     use_sahi=use_sahi, slice_size=slice_size, overlap=overlap)


if __name__ == "__main__":
    main()
