import numpy as np
import matplotlib.pyplot as plt
import torch
import io
import torchvision
import json

def plot_validation(img_t_, gts_, preds_, losses):
    plotted_imgs = []
    for i, img in enumerate(img_t_):
        _, H, W = img.size()
        img_np = img.permute(1, 2, 0).numpy()
        # Only get gt values that are not padded 
        gts_i = gts_[i, :, :]
        mask = gts_i[:, 2] != -1.0
        gts_i = gts_i[mask]
        # Prepare alphas to be a little bit nicer
        alphas = torch.sigmoid(preds_[i,:,2])
        # alphas = torch.where(preds_[i,:,2] > 0.8, 1.0, preds_[i,:,2])
        # alphas = torch.clamp(preds_[i,:,2], 0.1, 1.0)
        
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(img_np)

        # plot gts
        x_coords = gts_i[:, 0] * W
        y_coords = gts_i[:, 1] * H
        ax.scatter(x_coords, y_coords, c='green', s=20)
        # plot preds
        x_coords = preds_[i, :, 0] * W
        y_coords = preds_[i, :, 1] * H
        ax.scatter(x_coords, y_coords, c='red', marker='x', s=20, alpha=alphas)

        ax.axis('off')
        ax.text(5, 15, f'loss: {losses[i]:.2f}', fontsize = 12, c='white')
        
        # Convert plot to tensor with fixed size
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plot_img = plt.imread(buf)
        
        # Resize to consistent dimensions
        plot_tensor = torch.from_numpy(plot_img).permute(2, 0, 1).float()
        plot_tensor = torch.nn.functional.interpolate(
            plot_tensor.unsqueeze(0), size=(400, 400), mode='bilinear'
        ).squeeze(0)
        
        buf.close()
        plotted_imgs.append(plot_tensor)
        plt.close(fig)
    
    # Create grid of plotted images
    plotted_grid = torchvision.utils.make_grid(plotted_imgs)
    return plotted_grid


def plot_training_curve(epoch_train_losses, epoch_val_losses, fname):
    plt.plot(epoch_train_losses["total"], label="Train Total Loss")
    plt.plot(epoch_val_losses["total"], label="Val Total Loss")
    plt.plot(epoch_train_losses["cls"], label="Train cls Loss")
    plt.plot(epoch_val_losses["cls"], label="Val cls Loss")
    plt.plot(epoch_train_losses["reg"], label="Train reg Loss")
    plt.plot(epoch_val_losses["reg"], label="Val reg Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(fname)
    plt.close()

class LossTracker:
    def __init__(self):
        self.reset_epoch()
        self.epoch = {"total": [], "cls": [], "reg": []}

    def reset_epoch(self):
        self.running = {"total": 0.0, "cls": 0.0, "reg": 0.0}
        self.count = 0

    def update(self, losses):
        """Update running totals with (total, cls, reg)."""
        loss, cls, reg = losses
        self.running["total"] += loss
        self.running["cls"] += cls
        self.running["reg"] += reg
        self.count += 1

    @property
    def batch_avg(self):
        """Return averages so far in current epoch."""
        if self.count == 0:
            return {k: 0.0 for k in self.running}
        return {k: v / self.count for k, v in self.running.items()}

    def log_epoch(self, writer, split, epoch, save_path=None):
        """Compute epoch avg, save to history, log to TB."""
        epoch_avg = {k: v / self.count for k, v in self.running.items()}
        for k, v in epoch_avg.items():
            self.epoch[k].append(v)
        writer.add_scalars(f'Epoch {split} Loss', self.batch_avg, epoch)
        self.reset_epoch()

    def save(self, path):
        """Save all tracked losses to JSON"""
        with open(path, "w") as f:
            json.dump(self.epoch, f, indent=2)
