import torch
import yaml
import random
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


def predict_on_image(model, config, device, conf_threshold=0.5, plot_boxes=False):
    """Predict on a random image from validation dataset"""
    print("Running image prediction...")
    
    val_dataset = create_val_dataset_only()
    idx = random.randint(0, len(val_dataset) - 1)
    item = val_dataset[idx]
    
    pred_points, pred_scores = predict(model, item["img_t"], device, conf_threshold)
    
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
    pred_points_scaled = pred_points * config['img_size']  # Scale to image size
    ax.scatter(pred_points_scaled[:, 0], pred_points_scaled[:, 1], 
              marker='x', color='red', s=50)
    
    plt.title(f"Image Prediction (Found {len(pred_points)} points)")
    plt.show()


def predict_on_video(model, device, video_path, conf_threshold=0.5, start_frame=100):
    """Predict on video frames"""
    print("Running video prediction...")
    
    val_transform = CenterTransform(augment=False)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count < start_frame:
            continue
        
        # Convert BGR (OpenCV) → RGB (PIL expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Apply same preprocessing as dataloader
        image, *_ = val_transform(frame_pil, torch.empty((0, 2)))
        
        pred_points, pred_scores = predict(model, image, device, conf_threshold)
        
        # Scale predictions to frame dimensions
        H, W, _ = frame.shape
        pred_points_scaled = pred_points.copy()
        pred_points_scaled[:, 0] *= W
        pred_points_scaled[:, 1] *= H
        
        # Draw predictions on frame
        for x, y in pred_points_scaled:
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
        
        # Show frame info
        cv2.putText(frame, f"Frame: {frame_count}, Points: {len(pred_points)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Video prediction completed")


def main():
    # Load model once
    model, config, device = load_model_and_config()
    
    # Configuration
    video_path = "/home/tobias/data/gta2tv/processed/scenario_1/synced_camera_1.mp4"
    conf_threshold = 0.5
    
    
    # predict_on_image(model, config, device, conf_threshold, plot_boxes=False)
    predict_on_video(model, device, video_path, conf_threshold, start_frame=100)


if __name__ == "__main__":
    main()
