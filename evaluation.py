import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import yaml
from model import CenterPredictor
from dataset import get_val_dataset

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "experiments/BaselineDINO/"
IMG_SIZE = 224
AREA_THRESHOLDS = {
    'large': 200,
    'medium': 100
}

@dataclass
class EvalResult:
    """Container for evaluation results"""
    tp: int
    fp: int  
    fn: int
    assignments: List[Tuple]
    pred_scores: np.ndarray
    pred_categories: List[str]
    
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
    
    @property 
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

def get_box_category(box: np.ndarray) -> str:
    """Classify box as Small, Medium, or Large based on area"""
    xtl, ytl, xbr, ybr = box * IMG_SIZE
    area = (xbr - xtl) * (ybr - ytl)
    
    if area >= AREA_THRESHOLDS['large']:
        return 'L'
    elif area >= AREA_THRESHOLDS['medium']:
        return 'M'
    else:
        return 'S'

def match_predictions_to_gt(pred_points: np.ndarray, gt_boxes: np.ndarray, 
                           pred_scores: np.ndarray) -> EvalResult:
    """Match predictions to ground truth boxes using greedy assignment"""
    if len(pred_points) == 0:
        return EvalResult(0, 0, len(gt_boxes), [], pred_scores, [])
    if len(gt_boxes) == 0:
        return EvalResult(0, len(pred_points), 0, [], pred_scores, [None] * len(pred_points))
    
    # Get GT categories
    gt_categories = [get_box_category(box) for box in gt_boxes]
    
    matched_gt = set()
    assignments = [] #Pred_idx, gt_idx, matched
    tp_flags = np.zeros(len(pred_points), dtype=bool)
    pred_categories = [None] * len(pred_points)
    
    # Greedy matching - go through each prediction
    for pred_idx, (x, y) in enumerate(pred_points):
        match_found = False
        for gt_idx, (xtl, ytl, xbr, ybr) in enumerate(gt_boxes):
            # Already matched
            if gt_idx in matched_gt:
                continue
                
            # Check if prediction is inside GT box
            if xtl <= x <= xbr and ytl <= y <= ybr:
                # Match found
                matched_gt.add(gt_idx)
                assignments.append((pred_idx, gt_idx, True))
                tp_flags[pred_idx] = True
                pred_categories[pred_idx] = gt_categories[gt_idx]
                match_found = True
                break
        
        if not match_found:
            # No match found - this is a false positive
            assignments.append((pred_idx, None, False))
    
    tp = int(tp_flags.sum())
    fp = len(pred_points) - tp
    fn = len(gt_boxes) - len(matched_gt)
    
    return EvalResult(tp, fp, fn, assignments, pred_scores, pred_categories)

def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation"""
    # Add endpoints
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find recall threshold changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Compute AP
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

def compute_category_ap(all_scores: List[float], all_tp: List[bool], 
                      all_categories: List[str], n_gt: int, 
                      target_category: Optional[str] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute AP for specific category or overall"""
    if target_category is not None:
        # Filter for specific category
        mask = [(cat == target_category) or (not is_tp and cat is None) 
                for is_tp, cat in zip(all_tp, all_categories)]
        scores = np.array(all_scores)[mask]
        tp = np.array(all_tp)[mask]
    else:
        scores = np.array(all_scores)
        tp = np.array(all_tp)
    
    if len(scores) == 0:
        return 0.0, np.array([0]), np.array([0])
    
    # Sort by confidence
    sorted_idx = np.argsort(-scores)
    tp_sorted = tp[sorted_idx].astype(int)
    
    # Compute cumulative TP and FP
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)
    
    # Compute precision and recall
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / n_gt if n_gt > 0 else np.zeros_like(cum_tp)
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    ap = compute_ap_from_pr(precision, recall)
    return ap, recall, precision

def create_visualization(pred_points: np.ndarray, gt_boxes: np.ndarray, 
                        assignments: List[Tuple], image: torch.Tensor,
                        title: str = "") -> plt.Figure:
    """Create visualization of predictions and ground truth"""
    H, W = image.shape[1:]
    pred_px = pred_points * np.array([W, H])
    gt_centers = np.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                          (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2], axis=1)
    gt_centers_px = gt_centers * np.array([W, H])
    boxes_px = gt_boxes * np.array([W, H, W, H])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image.permute(1, 2, 0).cpu())
    
    # Draw GT boxes with category colors
    category_colors = {'S': 'green', 'M': 'yellow', 'L': 'red'}
    for box in gt_boxes:
        category = get_box_category(box)
        color = category_colors[category]
        
        box_px = box * np.array([W, H, W, H])
        xtl, ytl, xbr, ybr = box_px
        
        rect = patches.Rectangle(
            (xtl, ytl), xbr - xtl, ybr - ytl,
            linewidth=2, edgecolor=color, facecolor='none',
            label=f'{category} ({int((xbr-xtl)*(ybr-ytl))} px²)'
        )
        ax.add_patch(rect)
    
    # Plot GT centers and predictions
    ax.scatter(gt_centers_px[:, 0], gt_centers_px[:, 1], 
              c='blue', marker='o', s=60, label='GT Centers', alpha=0.8)
    ax.scatter(pred_px[:, 0], pred_px[:, 1], 
              c='red', marker='x', s=80, label='Predictions', alpha=0.8)
    
    # Draw assignment lines
    for pred_idx, gt_idx, is_match in assignments:
        if gt_idx is None:
            continue
        color = 'green' if is_match else 'orange'
        ax.plot([pred_px[pred_idx, 0], gt_centers_px[gt_idx, 0]],
               [pred_px[pred_idx, 1], gt_centers_px[gt_idx, 1]],
               color=color, linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_title(f"{title}\nTP: {sum(1 for _, _, match in assignments if match)}, "
                f"FP: {sum(1 for _, gt_idx, _ in assignments if gt_idx is None)}")
    ax.legend()
    ax.axis('off')
    
    return fig

class ObjectDetectionEvaluator:
    """Main evaluator class"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.val_dataset = None
        
    def load_model(self):
        """Load the trained model"""
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        self.model = CenterPredictor(
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
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path + 'best_model.pt', map_location=self.device)
        )
        self.model.eval()
        
    def load_dataset(self):
        """Load validation dataset"""
        self.val_dataset = get_val_dataset(img_size=IMG_SIZE)
    
    def predict(self, image: torch.Tensor, conf_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for a single image"""
        with torch.no_grad():
            preds = self.model(image.unsqueeze(0).to(self.device)).squeeze(0).cpu()
        
        # Apply sigmoid to confidence scores
        preds[:, 2] = torch.sigmoid(preds[:, 2])
        
        # Filter by confidence
        mask = preds[:, 2] > conf_threshold
        filtered_preds = preds[mask]
        
        points = filtered_preds[:, :2].numpy()
        scores = filtered_preds[:, 2].numpy()
        
        return points, scores
    
    def evaluate_precision_recall(self, conf_threshold: float = 0.5, max_samples: Optional[int] = None):
        """Evaluate precision and recall at fixed threshold"""
        if self.model is None:
            self.load_model()
        if self.val_dataset is None:
            self.load_dataset()
        
        results = []
        total_tp, total_fp, total_fn = 0, 0, 0
        
        samples = enumerate(self.val_dataset)
        if max_samples:
            samples = list(samples)[:max_samples]
        
        for i, item in tqdm(samples, desc=f"Evaluating P/R @ {conf_threshold}"):
            pred_points, pred_scores = self.predict(item["img_t"], conf_threshold)
            gt_boxes = item["boxes_t"].numpy()
            
            result = match_predictions_to_gt(pred_points, gt_boxes, pred_scores)
            results.append((result, item))
            
            total_tp += result.tp
            total_fp += result.fp  
            total_fn += result.fn
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        
        print(f"\nResults @ confidence {conf_threshold}:")
        print(f"Overall - TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"Precision: {overall_precision:.3f}, Recall: {overall_recall:.3f}")
        
        return results, overall_precision, overall_recall
    
    def evaluate_average_precision(self, max_samples: Optional[int] = None):
        """Evaluate Average Precision across all confidence thresholds"""
        if self.model is None:
            self.load_model()
        if self.val_dataset is None:
            self.load_dataset()
        
        all_scores, all_tp, all_categories = [], [], []
        category_gt_counts = {'S': 0, 'M': 0, 'L': 0}
        total_gt = 0
        
        samples = self.val_dataset
        if max_samples:
            samples = list(samples)[:max_samples]
        
        for item in tqdm(samples, desc="Computing AP"):
            pred_points, pred_scores = self.predict(item["img_t"], conf_threshold=0.0)
            gt_boxes = item["boxes_t"].numpy()
            
            result = match_predictions_to_gt(pred_points, gt_boxes, pred_scores)
            
            # Add scores and TP flags for each prediction
            all_scores.extend(result.pred_scores)
            
            # Create TP flags: True if prediction was matched (TP), False if unmatched (FP)
            tp_flags = np.zeros(len(result.pred_scores), dtype=bool)
            for pred_idx, gt_idx, is_match in result.assignments:
                tp_flags[pred_idx] = is_match  # This should already be True/False correctly
            
            all_tp.extend(tp_flags)
            all_categories.extend(result.pred_categories)
            
            # Count GT boxes by category
            for box in gt_boxes:
                cat = get_box_category(box)
                category_gt_counts[cat] += 1
                total_gt += 1
        
        # Compute AP for each category and overall
        ap_overall, recall_overall, precision_overall = compute_category_ap(
            all_scores, all_tp, all_categories, total_gt
        )
        
        results = {'overall': ap_overall}
        
        for cat in ['S', 'M', 'L']:
            if category_gt_counts[cat] > 0:
                ap_cat, _, _ = compute_category_ap(
                    all_scores, all_tp, all_categories, category_gt_counts[cat], cat
                )
                results[cat] = ap_cat
            else:
                results[cat] = 0.0
        
        print(f"\nAverage Precision Results:")
        print(f"Overall AP: {ap_overall:.3f}")
        print(f"AP Small: {results['S']:.3f} (n={category_gt_counts['S']})")
        print(f"AP Medium: {results['M']:.3f} (n={category_gt_counts['M']})")
        print(f"AP Large: {results['L']:.3f} (n={category_gt_counts['L']})")
        
        return results, (recall_overall, precision_overall)
    
    def save_visualizations(self, results: List, output_dir: str, n_best_worst: int = 2):
        """Save visualization plots for best/worst performing samples"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sort by precision and recall
        precisions = [r[0].precision for r in results]
        recalls = [r[0].recall for r in results]
        
        # Get indices for best/worst cases
        best_precision_idx = np.argsort(precisions)[-n_best_worst:]
        worst_precision_idx = np.argsort(precisions)[:n_best_worst]
        best_recall_idx = np.argsort(recalls)[-n_best_worst:]  
        worst_recall_idx = np.argsort(recalls)[:n_best_worst]
        
        cases = [
            (best_precision_idx, "best_precision"),
            (worst_precision_idx, "worst_precision"), 
            (best_recall_idx, "best_recall"),
            (worst_recall_idx, "worst_recall")
        ]
        
        for indices, name in cases:
            for i, idx in enumerate(indices):
                result, item = results[idx]
                pred_points, _ = self.predict(item["img_t"])
                gt_boxes = item["boxes_t"].numpy()
                
                fig = create_visualization(
                    pred_points, gt_boxes, result.assignments, item["img_t"],
                    f"{name}_{i} (P={result.precision:.2f}, R={result.recall:.2f})"
                )
                
                fig.savefig(f"{output_dir}/{name}_{i}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)

def main():
    """Main evaluation pipeline"""
    evaluator = ObjectDetectionEvaluator(MODEL_PATH, DEVICE)
    output_dir = MODEL_PATH + 'evaluation'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate Average Precision
    print("Computing Average Precision...")
    ap_results, (recall, precision) = evaluator.evaluate_average_precision()
    
    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.title(f'Precision-Recall Curve (AP={ap_results["overall"]:.3f})')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluate at fixed threshold
    print("\nEvaluating at fixed threshold...")
    pr_results, overall_p, overall_r = evaluator.evaluate_precision_recall(
        conf_threshold=0.5
    )
    
    # Save visualizations
    print("\nSaving visualizations...")
    evaluator.save_visualizations(pr_results, output_dir)
    
    # Save results to file
    with open(f'{output_dir}/results.txt', 'w') as f:
        f.write("=== EVALUATION RESULTS ===\n\n")
        f.write("Average Precision:\n")
        f.write(f"Overall AP: {ap_results['overall']:.3f}\n")
        f.write(f"AP Small: {ap_results['S']:.3f}\n") 
        f.write(f"AP Medium: {ap_results['M']:.3f}\n")
        f.write(f"AP Large: {ap_results['L']:.3f}\n\n")
        f.write("Precision/Recall @ 0.5:\n")
        f.write(f"Precision: {overall_p:.3f}\n")
        f.write(f"Recall: {overall_r:.3f}\n")
    
    print(f"\nEvaluation complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
