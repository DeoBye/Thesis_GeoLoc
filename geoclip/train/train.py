import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def add_gps_noise(gps, std_dev=0.01):
    """Injects Gaussian noise into GPS coordinates."""
    noise = torch.randn(gps.size()) * (150 / 111_320)                 #####
    return gps + noise

def train(train_dataloader, model, optimizer, epoch, batch_size, device, criterion, scheduler=None, gps_noise_std=0.01, logger=None):
    

    model.train()
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    # targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)

    for i ,(imgs, gps) in bar:
        imgs = imgs.to(device)
        gps = add_gps_noise(gps, std_dev=gps_noise_std)
        gps = gps.to(device)
        optimizer.zero_grad()
        # gps_queue = model.get_gps_queue() 
        # Append GPS Queue & Queue Update
        # gps_all = torch.cat([gps, gps_queue], dim=0)
        # logits_img_gps = model(imgs, gps_all)
        # img_gps_loss = criterion(logits_img_gps, targets_img_gps)
        
        
        loss = criterion(imgs, gps, batch_size)
        # model.dequeue_and_enqueue(gps)     
        
        # Backpropagate
        loss.backward()
        optimizer.step()

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
        
        if logger:
            logger.info(f"Batch {i + 1}/{len(train_dataloader)}: Loss = {loss.item():.5f}")
    if scheduler is not None:
        scheduler.step()
