import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from .loss import SimsiamLoss, get_loss

def add_gps_noise(gps, std_dev=0.01):
    """Injects Gaussian noise into GPS coordinates."""
    noise = torch.randn(gps.size()) * (150 / 111_320)                 #####
    # noise = torch.randn(gps.size()) * std_dev
    return gps + noise


def train(train_dataloader, model, optimizer, epoch, strategy, strategy_name, scheduler=None, logger=None):
    

    model.train()
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    # targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)

    for i, items in bar:
        
        optimizer.zero_grad()        
        outputs = strategy(items)
        loss_dict = get_loss(outputs, strategy_name)           
        # Backpropagate
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
        
        if logger:
            if strategy_name == "ssl":
                logger.info(f"Batch {i + 1}/{len(train_dataloader)}: Loss = {loss_dict['loss'].item():.5f}, Clip Loss = {loss_dict['clip_loss'].item():.5f}, simsiam Loss = {loss_dict['simsiam_loss'].item():.5f}")
            else:
                logger.info(f"Batch {i + 1}/{len(train_dataloader)}: Loss = {loss_dict['loss'].item():.5f}")
    if scheduler is not None:
        scheduler.step()
        
        
def train_ssl(train_dataloader, model, optimizer, epoch, batch_size, device, criterion_two, simsiam_criterion, scheduler=None, gps_noise_std=0.01, logger=None):
    

    model.train()
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    # targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(device)

    for i ,(imgs, gps, aug1, aug2) in bar:
        # imgs = imgs.to(device)
        aug1 = aug1.to(device)
        aug2 = aug2.to(device)
        # gps = add_gps_noise(gps, std_dev=gps_noise_std)
        gps = gps.to(device)
        optimizer.zero_grad()
        # gps_queue = model.get_gps_queue() 
        # Append GPS Queue & Queue Update
        # gps_all = torch.cat([gps, gps_queue], dim=0)
        # logits_img_gps = model(imgs, gps_all)
        # img_gps_loss = criterion(logits_img_gps, targets_img_gps)
        aug_features1 = model.image_encoder(aug1)
        aug_features2 = model.image_encoder(aug2)
        z1 = model.projector(aug_features1)
        z2 = model.projector(aug_features2)
        p1 = model.predictor(z1)
        p2 = model.predictor(z2)

        simsiam_loss = simsiam_criterion(p1, z1, p2, z2)
        clip_loss = criterion_two(aug1, aug2, gps, batch_size)
        
        
        # model.dequeue_and_enqueue(gps)     
        
        # Backpropagate
        loss = 0.9 * clip_loss + 0.1 * simsiam_loss
        loss.backward()
        optimizer.step()

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
        
        if logger:
            logger.info(f"Batch {i + 1}/{len(train_dataloader)}: Loss = {loss.item():.5f}, Clip Loss = {clip_loss.item():.5f}, simsiam Loss = {simsiam_loss.item():.5f}")
    if scheduler is not None:
        scheduler.step()
