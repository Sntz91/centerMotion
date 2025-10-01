import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np


def get_cost_matrix(gt_centers, pred_centers, pred_obj_logits, center_weight=5.0, objectness_weight=2.0):
    cost_center = torch.cdist(gt_centers, pred_centers, p=1)    
    objectness = pred_obj_logits.sigmoid().unsqueeze(0)     
    # cost_class = -alpha * objectness.clamp(min=1e-9).log()
    # cost_class = cost_class.expand(gt_centers.size(0), -1)
    # TODO: if objectness approaches 0, we have a problem here. 
    cost_class = -objectness.log() #- (1 - objectness).log() #CHANGED THIS
    cost_matrix = (objectness_weight * cost_class + center_weight * cost_center)
    return cost_matrix

def binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)  
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_factor * (1 - p_t) ** gamma
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = focal_weight * bce
    return loss.sum()

def center_loss_fn(preds, gts, gt_lengths, center_weight=2.0, objectness_weight=1.0, focal_alpha=0.25, focal_gamma=2.0, cost_center_weight=5.0, cost_objectness_weight=2.0):
    pred_centers = preds[:,:,:2]
    pred_logits = preds[:,:,2]
    gt_centers = gts[:,:,:2]
    B, N, _ = pred_centers.shape
    device = pred_centers.device

    total_center_loss = 0.0
    total_obj_loss = 0.0
    total_gt = 0

    # Iterate over Batch
    for b in range(B):
        num_gt = gt_lengths[b].item()
        # No Objects: everything should predict 0
        if num_gt == 0:
            obj_loss = F.binary_cross_entropy_with_logits(
                pred_logits[b], torch.zeros_like(pred_logits[b]), reduction='mean'
            )
            total_obj_loss += obj_loss
            continue
          
        gt_c = gt_centers[b, :num_gt] # For matching we only use real gts!
        pred_c = pred_centers[b]
        pred_l = pred_logits[b]

        # Cost & Matching
        with torch.no_grad():
            cost_matrix = get_cost_matrix(gt_c, pred_c, pred_l, cost_center_weight, cost_objectness_weight)
            gt_idx, pred_idx = linear_sum_assignment(cost_matrix.cpu().numpy())

        # Matched pairs
        matched_gt = gt_c[gt_idx] 
        matched_pred = pred_c[pred_idx] 
        matched_logits = pred_l[pred_idx]

        # Center L1 loss
        center_loss = F.l1_loss(matched_pred, matched_gt, reduction='sum')
        total_center_loss += center_loss

        # Classification Loss
        targets = torch.zeros_like(pred_l)
        targets[pred_idx] = 1.0 
        # probs = pred_l.sigmoid()
        # ce_loss = F.binary_cross_entropy_with_logits(pred_l, targets, reduction='sum')
        # ce_loss = binary_focal_loss(pred_l, targets, alpha=0.25, gamma=2.0, reduction='sum')
        ce_loss = binary_focal_loss(pred_l, targets, alpha=focal_alpha, gamma=focal_gamma)
        total_obj_loss += ce_loss

        total_gt += num_gt

    # Normalize
    center_loss = total_center_loss / max(total_gt, 1)
    obj_loss = total_obj_loss / max(total_gt, 1)#(B*N) #CHANGED THIS

    total_loss = center_weight * center_loss + objectness_weight * obj_loss
    return total_loss, obj_loss, center_loss


if __name__ == '__main__':
    # preds: B, N, 3
    # GT: B, N, 3
    preds = torch.tensor([
        [[1.0, 1.0, 100],
         [2.0, 2.0, 100],
         [0.0, 0.0, -100],
         [0.0, 0.0, -100]],
        [[4.0, 4.0, 100],
         [5.0, 5.0, 100],
         [6.0, 6.0, 100],
         [0.0, 0.0, -100]]
    ])
    gts = torch.tensor([
        [[1.0, 1.0, 1.0],
         [2.0, 2.0, 1.0],
         [3.0, 3.0, 1.0]],
        [[4.0, 4.0, 1.0],
         [5.0, 5.0, 1.0],
         [6.0, 6.0, 1.0]]
    ])
    gt_lengths = torch.tensor([3, 3])

    loss, _, _, _ = center_loss_fn(preds, gts, gt_lengths)
    print(loss)
