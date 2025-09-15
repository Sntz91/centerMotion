import torch
import torchvision 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler
from model.model import initialize_model_from_config
from data.dataset import prepare_dataset_from_config
from utils.utils import plot_validation, plot_training_curve, LossTracker
from model.loss import center_loss_fn
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import yaml
import shutil
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def log_predictions(model, batch, writer, step, tag="Train"):
    model.eval()
    preds = model(batch["img_t"].to(DEVICE)).cpu()
    losses = [center_loss_fn(pred.unsqueeze(0), gt.unsqueeze(0), length.unsqueeze(0))[0].item() for pred, gt, length in zip(preds, batch["gt_t"], batch["lengths_t"])]
    plotted_grid = plot_validation(batch["img_t"], batch["gt_t"], preds, losses)
    writer.add_image(f'Plotted Images Grid {tag}', plotted_grid, step)

def training_step(model, batch, optimizer, scaler):
    model.train()
    with autocast():
        preds = model(batch["img_t"].to(DEVICE))
    # SO DO I CALC LOSS ON CPU OR GPU HERE? BOTH. hmm...
        loss, cls_loss, reg_loss = center_loss_fn(preds, batch["gt_t"].to(DEVICE), batch["lengths_t"].to(DEVICE))
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), cls_loss.item(), reg_loss.item()

def validation_step(model, batch):
    model.eval()
    with torch.no_grad():
        preds = model(batch["img_t"].to(DEVICE))
        loss, cls_loss, reg_loss = center_loss_fn(preds, batch["gt_t"].to(DEVICE), batch["lengths_t"].to(DEVICE))
    return loss.item(), cls_loss.item(), reg_loss.item()

def main():
    # INITIALIZATION
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_name = config["model_name"]
    os.makedirs(f'experiments/{model_name}', exist_ok=False)
    shutil.copy2('config.yaml', f'experiments/{model_name}/config.yaml')
    writer = SummaryWriter(log_dir=f"experiments/{model_name}/logs")
    
    train_dataloader, val_dataloader = prepare_dataset_from_config(config)

    model = initialize_model_from_config(config).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["lr"], 
        weight_decay=config["weight_decay"]
    )
    scaler = GradScaler()
    # profile_batches = 20
    # prof = profile(
        # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes=True,
        # with_stack=True,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    # ) 
    best_val_loss = torch.inf
    val_step = 0
    train_step = 0

    train_tracker = LossTracker()
    val_tracker = LossTracker()
    
    # LETS GO
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch+1}/{config['epochs']}")

        # --- Training ---
        # prof.start()
        pbar = tqdm(train_dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            train_step+=1
            # if batch_idx < profile_batches:
                # prof.step()
            loss_tuple = training_step(model, batch, optimizer, scaler)
            train_tracker.update(loss_tuple)
            pbar.set_postfix(train_tracker.batch_avg)
            if batch_idx%50==0:
                writer.add_scalars(f'TRAIN Loss', train_tracker.batch_avg, train_step)
            if batch_idx%300==0:
                log_predictions(model, batch, writer, train_step, "TRAIN")
            # if batch_idx >= profile_batches:
                # prof.stop()

        # --- Validation ---
        draw_choice = random.randint(0, len(val_dataloader)-1)
        pbar = tqdm(val_dataloader, desc="Validation")
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                val_step+=1
                draw = True if batch_idx == draw_choice else False
                loss_tuple = validation_step(model, batch)
                val_tracker.update(loss_tuple)
                pbar.set_postfix(val_tracker.batch_avg)
                if draw:
                    log_predictions(model, batch, writer, val_step, "VAL")

        # Log loss
        train_tracker.log_epoch(writer, "train", epoch)
        val_tracker.log_epoch(writer, "val", epoch)
    
        # --- Checkpoints ---
        if val_tracker.epoch["total"][-1] < best_val_loss:
            best_val_loss = val_tracker.epoch["total"][-1]
            torch.save(model.state_dict(), f"experiments/{model_name}/best_model.pt")
            print("🔸 Saved new best model.")

    torch.save(model.state_dict(), f"experiments/{model_name}/checkpoint_final_epoch.pt") 
    writer.close()

    # Save Loss curve 
    plot_training_curve(train_tracker.epoch, val_tracker.epoch, f'experiments/{model_name}/training_curve.png')
    train_tracker.save(f"experiments/{model_name}/train_losses.json")
    val_tracker.save(f"experiments/{model_name}/val_losses.json")


if __name__ == '__main__': 
    main()
