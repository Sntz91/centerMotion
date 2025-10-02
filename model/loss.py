import torch
import torch.nn as nn

class GaussianCenterLoss(nn.Module):
    def __init__(self, k_factor=0.5, l1_weight_schedule=None, num_epochs=10):
        super().__init__()
        self.k_factor = k_factor
        # Use l1_weight_schedule for the weight of the L1 term
        # This will be a float, e.g., 1.0
        self.l1_weight_schedule = l1_weight_schedule if l1_weight_schedule is not None else 0.0
        self.num_epochs = num_epochs
        self.current_epoch = 0 # Remember to update this externally!

    def get_max_heatmap_value(self, query_xy, centers, widths, heights):
        diff_x_sq = (query_xy[:, 0:1] - centers[:, 0].unsqueeze(0))**2 # (N, M)
        diff_y_sq = (query_xy[:, 1:2] - centers[:, 1].unsqueeze(0))**2 # (N, M)
        # print('diff_x_sq', diff_x_sq)
        # print('diff_y_sq', diff_x_sq)
        # decrease std over time while decreasing l1 also
        sigma_x_sq = (self.k_factor * (widths*(1.0+current_l1_weight*2.0))) ** 2
        sigma_y_sq = (self.k_factor * (heights*(1.0+current_l1_weight*2.0))) ** 2

        sigma_x_sq = sigma_x_sq.unsqueeze(0) # Now (1, M)
        sigma_y_sq = sigma_y_sq.unsqueeze(0) # Now (1, M)
        # print('sigma_x_sq', sigma_x_sq)
        # print('sigma_y_sq', sigma_y_sq)
        
        # Division is now (N, M) / (1, M) -> (N, M)
        exponent = -(diff_x_sq / (2 * sigma_x_sq + 1e-9) + diff_y_sq / (2 * sigma_y_sq + 1e-9))
        # scaled_sq_error = (diff_x_sq / (2 * sigma_x_sq + 1e-9) + diff_y_sq / (2 * sigma_y_sq + 1e-9)) # (N, M)
        # min_error, _ = torch.min(scaled_sq_error, dim=1) # (N,)
        # print('exponent', exponent)
        gaussians = torch.exp(exponent)  # (N,M)
        # print('gaussians', gaussians)

        max_intensity, _ = torch.max(gaussians, dim=1)  # max over centers
        # print('max_intensity', max_intensity)
        # print('return value', 1.0 - max_intensity)

        return 1.0 - max_intensity  # (N,) #min_error


    def get_min_distance_loss(self, pred_xy, gt_cxcy):
        """
        Calculates the L1 distance from each prediction to the closest GT center.
        Args:
            pred_xy: (N, 2) predicted centers
            gt_cxcy: (M, 2) GT centers
        Returns:
            loss: (N,) tensor of minimum L1 distances
        """
        N, _ = pred_xy.shape
        M, _ = gt_cxcy.shape

        if M == 0:
            # If no GT objects, the L1 loss is 0 for all predictions
            return torch.zeros(N, device=pred_xy.device)
        
        # Calculate L1 distance: |px - cx| + |py - cy|
        # pred_xy[:, None, :] gives (N, 1, 2)
        # gt_cxcy[None, :, :] gives (1, M, 2)
        # diff_abs: (N, M, 2)
        diff_abs = torch.abs(pred_xy[:, None, :] - gt_cxcy[None, :, :])
        
        # l1_distances: (N, M)
        l1_distances = diff_abs.sum(dim=-1)
        
        # Find the minimum L1 distance for each prediction: (N,)
        min_l1_distance, _ = torch.min(l1_distances, dim=1)
        
        return min_l1_distance

    def forward(self, preds, gt_boxes):
        B, N, _ = preds.shape
        device = preds.device
        total_loss = 0.0
        current_l1_weight = self.get_l1_weight() # Get the scheduled weight
        
        for b in range(B):
            # print('---' * 10, 'batch', '---' * 10)
            pred_xy = preds[b, :, :2]
            box = gt_boxes[b].float().to(device)
            gt_cxcy = box[:, :2]
            gt_widths = box[:, 2]
            gt_heights = box[:, 3]
            num_pred = pred_xy.shape[0]
   
            # TODO: The loss cant handle the fact that there might be no GT
            num_gt_objects = gt_cxcy.shape[0]
            if num_gt_objects == 0:
                print('NO GT FOUND!')
                batch_loss = torch.tensor(0.0, device=device)
                total_loss += batch_loss
                continue
            
            # --- 1. Pred -> GT Gaussian Loss (Precision) ---
            pred_gaussian_loss = self.get_max_heatmap_value(
                pred_xy, gt_cxcy, gt_widths, gt_heights
            )
            print('pred->gt', pred_gaussian_loss)
            
            # --- 2. GT -> Pred Gaussian Loss (Coverage) ---
            # Keep your original large sigma for a smooth coverage signal
            gt_gaussian_loss = self.get_max_heatmap_value(
                gt_cxcy, pred_xy,
                torch.full((num_pred,), 0.2, device=device), 
                torch.full((num_pred,), 0.2, device=device)  
            )
            print('coverage', gt_gaussian_loss)
            
            # --- 3. L1 Regression Loss (Exploration/Guidance) ---
            l1_loss = self.get_min_distance_loss(pred_xy, gt_cxcy)
            print('l1_loss', l1_loss)
            # print('gt_gaussian_loss', gt_gaussian_loss)
            # print('pred_gaussian_loss', pred_gaussian_loss)

            # Combine all losses
            batch_loss = (
                pred_gaussian_loss.sum() + 
                gt_gaussian_loss.sum() + 
                current_l1_weight * l1_loss.sum()
            )
            print('batch_loss', batch_loss)
            
            total_loss += batch_loss
        print('total_loss', total_loss)
        return total_loss

    def get_l1_weight(self):
        """
        Calculates the L1 weight based on the current epoch using a simple annealing schedule.
        Start high, end low (or 0).
        """
        # A simple linear decay schedule: L1 weight goes from initial value down to 0.0
        decay_factor = max(0.0, 1.0 - (self.current_epoch / self.num_epochs))
        return self.l1_weight_schedule * decay_factor

    def set_epoch(self, epoch):
        """
        Call this at the beginning of each training epoch.
        """
        self.current_epoch = epoch


def visualize_loss_component(ax, grid_vals, title, img, gt_cxcy_coords, preds_coords, grid_size=518):
    """Helper function to plot a single loss component."""
    # Rescale grid values to the image size for plotting
    plot_vals = grid_vals.numpy().reshape(grid_size, grid_size)
    
    # Background image
    ax.imshow(img) 
    im = ax.imshow(plot_vals, cmap='jet', alpha=0.5, origin='lower') 
    
    # Overlay GT centers (denormalized coordinates)
    ax.scatter(gt_cxcy_coords[:, 0], gt_cxcy_coords[:, 1], 
               color='white', edgecolors='black', marker='o', s=100, label='GT Centers')
    
    # Overlay Example Predictions (denormalized coordinates)
    ax.scatter(preds_coords[0, :, 0].numpy(), preds_coords[0, :, 1].numpy(), 
               color='cyan', edgecolors='blue', marker='x', s=150, linewidths=2, label='Example Preds')

    ax.set_title(title)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


if __name__ == '__main__':
    import yaml 
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from data.dataset import create_val_dataset_only

    # --- Configuration ---
    TARGET_EPOCH = 10
    GRID_SIZE = 518
    
    # --- Load data/loss ---
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    # train_dataloader, val_dataloader = prepare_dataset_from_config(config)
    val_dataset = create_val_dataset_only(img_size=518)
    batch = val_dataset[1000]

    loss_fn = GaussianCenterLoss(k_factor=0.3, l1_weight_schedule=1.0, num_epochs=10)
    loss_fn.set_epoch(TARGET_EPOCH)
    
    current_l1_weight = loss_fn.get_l1_weight()
    print(f"Visualizing for Epoch {TARGET_EPOCH}. L1 Weight = {current_l1_weight:.4f}")

    # --- Grab one batch ---
    # for batch in val_dataloader:
    img = batch['img_t'].permute(1, 2, 0).numpy()  # [H,W,C]
    gt_boxes = batch['boxes_t']  # (M,4)
    
    # Example predictions
    preds = torch.tensor([
        [425.2, 425],
        [424, 423],
        [412, 20],
        [382, 32.3],
        [382, 32.3],
        [382, 32.3],
        [0,0]
    ]).unsqueeze(0) / GRID_SIZE
    """preds = torch.tensor([
        [0.0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]).unsqueeze(0) / GRID_SIZE"""
    # preds = gt_boxes[:, :2].unsqueeze(0) * 0
    # preds = torch.rand(preds.shape)
    print('preds', preds)
    
    gt_cxcy = gt_boxes[:, :2]  
    gt_wh = gt_boxes[:, 2:] 
    
    # Denormalized coords
    gt_cxcy_coords_denorm = gt_cxcy * GRID_SIZE
    preds_coords_denorm = preds * GRID_SIZE
        # break
    
    loss = loss_fn(preds, gt_boxes.unsqueeze(0))
    print('loss', loss)

    # --- Build grid for visualization ---
    xs = np.linspace(0, 1, GRID_SIZE)
    ys = np.linspace(0, 1, GRID_SIZE)
    X, Y = np.meshgrid(xs, ys)
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

    # --- Compute loss terms ---
    with torch.no_grad():
        # 1. Pred→GT Gaussian (precision)
        gaussian_loss_vals = loss_fn.get_max_heatmap_value(
            grid_points, gt_cxcy, gt_wh[:, 0], gt_wh[:, 1]
        )
        # print('pred->gt', gaussian_loss_vals)
        
        # 2. L1 global guide
        l1_loss_vals = loss_fn.get_min_distance_loss(grid_points, gt_cxcy)
        num_pred = preds.shape[1]
        
        # 3. GT→Pred Gaussian (coverage)
        coverage_loss_vals = loss_fn.get_max_heatmap_value(
            grid_points, preds[0], 
                torch.full((num_pred,), 0.2),
                torch.full((num_pred,), 0.2)  
        )
        # print('coverage', coverage_loss_vals.mean())
        
        # 4. Total hybrid loss
        total_loss_vals = gaussian_loss_vals + coverage_loss_vals + current_l1_weight * l1_loss_vals
        # print('total', total_loss_vals.mean())

    # --- Plot all components ---
    fig, axes = plt.subplots(1, 4, figsize=(28, 6), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.3)

    # Plot 1: Pred→GT Gaussian
    visualize_loss_component(
        axes[0], gaussian_loss_vals, 
        '1. Pred→GT Gaussian (Precision)', 
        img, gt_cxcy_coords_denorm, preds_coords_denorm, GRID_SIZE
    )

    # Plot 2: L1 Loss
    visualize_loss_component(
        axes[1], l1_loss_vals, 
        f'2. L1 Loss (Guide, Weight={current_l1_weight:.2f})', 
        img, gt_cxcy_coords_denorm, preds_coords_denorm, GRID_SIZE
    )

    # Plot 3: GT→Pred Gaussian
    visualize_loss_component(
        axes[2], coverage_loss_vals, 
        '3. GT→Pred Gaussian (Coverage)', 
        img, gt_cxcy_coords_denorm, preds_coords_denorm, GRID_SIZE
    )

    # Plot 4: Total hybrid loss
    visualize_loss_component(
        axes[3], total_loss_vals, 
        '4. Total Hybrid Loss (All Terms)', 
        img, gt_cxcy_coords_denorm, preds_coords_denorm, GRID_SIZE
    )

    plt.tight_layout()
    plt.show()
