import torch
import torch.nn as nn

class GaussianCenterLoss(nn.Module):
    def __init__(self, k_factor=0.5, temp=0.2, img_size=518, sigma_start=0.5, sigma_end=None, sigma_decay_epochs=10):
        super().__init__()
        self.k_factor = k_factor
        self.temp = temp
        self.img_size = img_size
        self.sigma_start = sigma_start        # initial wide Gaussian
        self.sigma_end = sigma_end or k_factor  # target Gaussian (scaled by object size)
        self.sigma_decay_epochs = sigma_decay_epochs
        self.current_sigma = sigma_start
        self.epoch = 0

    def update_sigma(self, epoch):
        self.epoch = epoch
        t = min(epoch / self.sigma_decay_epochs, 1.0)
        # linear decay
        self.current_sigma = (1-t)*self.sigma_start + t*self.sigma_end


    def get_max_heatmap_value(self, query_xy, centers, widths, heights):
        # Broadcast differences: (N,M)
        N, _ = query_xy.shape
        M = centers.shape[0]
        if M == 0:
            return torch.zeros(N, device=query_xy.device)

        diff_x_sq = (query_xy[:, 0:1] - centers[:, 0].unsqueeze(0))**2
        diff_y_sq = (query_xy[:, 1:2] - centers[:, 1].unsqueeze(0))**2

        sigma_x_sq = torch.clamp(self.k_factor * widths, min=self.current_sigma) ** 2
        sigma_y_sq = torch.clamp(self.k_factor * heights, min=self.current_sigma) ** 2

        exponent = -(diff_x_sq / (2 * sigma_x_sq + 1e-9) + diff_y_sq / (2 * sigma_y_sq + 1e-9))
        gaussians = torch.exp(exponent)  # (N,M)

        # max_intensity, _ = torch.max(gaussians, dim=1)  # max over centers
        # softmax better
        softmax_weights = torch.softmax(gaussians / self.temp, dim=1)
        max_intensity = (gaussians * softmax_weights).sum(dim=1)

        return 1.0 - max_intensity  # (N,)

    def forward(self, preds, gt_boxes):
        """
        Args:
            preds: (B, N, 2) predicted normalized centers [x, y] in [0,1]
            gt_boxes: (B, M, 4) GT boxes [cx, cy, w, h] normalized to [0,1]

        Returns:
            loss: scalar tensor
        """
        B, N, _ = preds.shape
        device = preds.device

        total_loss = 0.0

        for b in range(B):
            pred_xy = preds[b]
            box = gt_boxes[b].to(device)
            gt_cxcy = box[:, :2]
            gt_widths = box[:, 2]
            gt_heights = box[:, 3]
            num_pred = pred_xy.shape[0]

            # -------------------------------
            # 1. Pred->GT loss
            # -------------------------------
            pred_loss = self.get_max_heatmap_value(
                pred_xy, 
                gt_cxcy, 
                gt_widths, 
                gt_heights
            )

            # -------------------------------
            # 2. GT->Pred (coverage) loss
            # -------------------------------
            # this kind of depends on how many gts you have. we can normalize this later TODO
            gt_loss = self.get_max_heatmap_value(
                gt_cxcy, 
                pred_xy, 
                torch.full((num_pred,), self.current_sigma, device=device),
                torch.full((num_pred,), self.current_sigma, device=device)
            )
            
            # We take mean instead of sum to make it independent of nr of elements
            total_loss += pred_loss.mean() + gt_loss.mean()

        return total_loss


if __name__ == '__main__':
    """
    from data.dataset import prepare_dataset_from_config
    import yaml

    loss_fn = GaussianCenterLoss(k_factor=0.3)
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    train_dataloader, val_dataloader = prepare_dataset_from_config(config)
    for batch in train_dataloader:
        lengths = batch['lengths_t']
        gt_boxes = batch['boxes_t'] 
        # print(batch['boxes_t'])
        print(lengths)
        preds = torch.tensor([
            [425.2, 425],
            [382, 32.3],
        ]).unsqueeze(0)/518
        loss = loss_fn(preds, gt_boxes)
        print(loss)
        break
    """
    import torch
    import matplotlib.pyplot as plt
    import yaml
    from data.dataset import prepare_dataset_from_config
    from model.loss import GaussianCenterLoss
    import numpy as np

    # --- Load dataset ---
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    train_dataloader, val_dataloader = prepare_dataset_from_config(config)

    # --- Initialize loss ---
    loss_fn = GaussianCenterLoss(k_factor=0.3, temp=0.1, img_size=518)

    # --- Grab one batch ---
    for batch in train_dataloader:
        img = batch['img_t'][0].permute(1,2,0).numpy()  # [H,W,C] for matplotlib
        gt_boxes = batch['boxes_t'][0]  # (M,4)
        num_gt = gt_boxes.shape[0]

        # Example predicted centers (normalized)
        preds = torch.tensor([
            [425.2, 425],
            [382, 32.3],
        ]).unsqueeze(0)/518

        # Convert GT boxes to normalized centers/widths/heights
        gt_cxcy = gt_boxes[:, :2] / 518
        gt_wh = gt_boxes[:, 2:] / 518

        break  # just use first batch

    # --- Create grid of prediction points ---
    grid_size = 200
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(xs, ys)
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

    # --- Compute Pred->GT loss at each grid point ---
    with torch.no_grad():
        loss_vals = loss_fn.get_max_heatmap_value(grid_points, gt_cxcy, gt_wh[:,0], gt_wh[:,1])
        loss_vals = loss_vals.numpy().reshape(grid_size, grid_size)

    # --- Plot heatmap on top of the image ---
    plt.figure(figsize=(8,8))
    plt.imshow(img)  # background image
    plt.imshow(loss_vals, extent=[0,1,0,1], origin='lower', cmap='viridis', alpha=0.5)
    plt.colorbar(label='Pred->GT Loss')
    plt.title('GaussianCenterLoss Heatmap on Real Image')
    plt.xlabel('Normalized x')
    plt.ylabel('Normalized y')

    # Optionally overlay GT centers
    plt.scatter(gt_cxcy[:,0], gt_cxcy[:,1], c='red', marker='x', s=100, label='GT Centers')
    plt.legend()
    plt.show()
    # loss_fn = GaussianCenterLoss(k_factor=0.3)
    # val_dataset = create_val_dataset_only(img_size=512)
    # item = val_dataset[1000]
    # print(item['boxes_t'])
    # gt_boxes = item['boxes_t'].unsqueeze(0)

    # preds = torch.tensor([
        # [425.2, 425],
        # [382, 32.3],
    # ]).unsqueeze(0)/518
    # preds = gt_boxes[0, :, :2].unsqueeze(0)

    # loss = loss_fn(preds, gt_boxes)
    # print(loss)
