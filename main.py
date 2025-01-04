import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from geoclip import GeoCLIP, GeoDataLoader, GeoDataLoader_im2gps, GeoDataLoader_osm5v
from geoclip import img_train_transform, img_val_transform, train, eval_images
from geoclip.utils.logger import *
from geoclip import CrossEntropyLoss, TripletLoss, EntropywithDis

###  before run: main中改experiment path， log名字   loss_fn      geoclip.py中改loss_fn    loss中改



def add_gps_noise(gps, std_dev=0.01):
    """Injects Gaussian noise into GPS coordinates."""
    noise = torch.randn(gps.size()) * std_dev
    return gps + noise

# Training function for multiple epochs


# Function to save model checkpoint
def save_checkpoint(model, save_dir):
    """Saves the model checkpoint."""
    torch.save(model.image_encoder.mlp.state_dict(), os.path.join(save_dir, "image_encoder_mlp_weights.pth"))
    torch.save(model.location_encoder.state_dict(), os.path.join(save_dir, "location_encoder_weights.pth"))
    torch.save(model.logit_scale.data, os.path.join(save_dir, "logit_scale_weights.pth"))


def get_loss(loss_name, **kwargs):
    if loss_name == "crossentropy":
        return CrossEntropyLoss(**kwargs)
    if loss_name == "triplet":
        return TripletLoss(**kwargs)
    if loss_name == "entropywithdis":
        return EntropywithDis(**kwargs)

# Main script for training and validating GeoCLIP
def main(validate_only=False, test_only=False):
    # Hyperparameters
    experiment_path = "./experiments/"                              #####
    os.makedirs(experiment_path, exist_ok=True)
    
    log_file = os.path.join(experiment_path, "logger.log")                   #####
    logger = get_root_logger(log_file=log_file)
    
    num_epochs = 20
    batch_size = 128
    learning_rate = 3e-5
    weight_decay = 1e-6
    
    gps_noise_std = 0.01  # Standard deviation for GPS noise
    # model_save_path = "geoclip.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    #####
    # device = torch.device("cpu")
    print(device)

    # Initialize model, optimizer, and scheduler
    model = GeoCLIP(from_pretrained=True, queue_size=4096).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.87)
    
    loss_fn = 'entropywithdis'                             ####      if crossentropy: geoclip
    if loss_fn == 'crossentropy':
        criterion = get_loss('crossentropy', model=model, device=device)
    elif loss_fn == 'triplet':
        criterion = get_loss('triplet', model=model, device=device)
    elif loss_fn == "entropywithdis":
        criterion = get_loss('entropywithdis', model=model, device=device)
    
    print_log(f"Loss Function: {loss_fn}", logger)
    
    # Prepare data loaders
    train_dataset = GeoDataLoader("/root/global-streetscapes/data/map/map_filter.csv", "/root/global-streetscapes/data/map/images", transform=img_train_transform())
    # val_dataset = GeoDataLoader("/root/val.csv", "/root/data", transform=img_val_transform())
    val_dataset = GeoDataLoader("/root/global-streetscapes/data/map/val.csv", "/root/global-streetscapes/data/map/images", transform=img_val_transform())
    # test_dataset = GeoDataLoader_im2gps("/root/Im2GPS3K/im2gps3k_places365.csv", "/root/Im2GPS3K/im2gps3k_rgb_images", transform=img_val_transform())
    test_dataset = GeoDataLoader_osm5v("/root/osm5v/filtered.csv", "/root/osm5v/00", transform=img_val_transform())
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    best_acc = 0.0
    
    if validate_only:
    # Only validate the dataset
        print_log("Validating the model...", logger)
        val_loss, val_accuracy = eval_images(val_dataloader, model, device)
        print_log(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}", logger)
        return
    # Training and validation loop

    
    if test_only:
        print_log("Test the model...", logger)
        val_loss = eval_images(test_dataloader, model, device=device, logger=logger)
        print_log(f"Test Loss: {val_loss:.4f}, Test Accuracy: {val_accuracy:.4f}", logger)
        return
    
    print_log("starting training ...", logger)
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        print_log(f"Epoch {epoch}/{num_epochs}:", logger)
        epoch_dir = os.path.join(experiment_path, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        train(train_dataloader, model, optimizer, epoch, batch_size, device, criterion, scheduler, gps_noise_std=gps_noise_std, logger=logger)
        
        # Validate after each epoch
        print_log("validate: ...", logger)
        val_acc = eval_images(val_dataloader, model, device=device, logger=logger)
        print_log("test on im2gps3k: ...", logger)
        test_acc = eval_images(test_dataloader, model, device=device, logger=logger)
        # Save the model checkpoint
        save_checkpoint(model, save_dir=epoch_dir)

        print_log(f"Epoch {epoch} completed. Model saved to {epoch_dir}.", logger)


if __name__ == "__main__":
    main(validate_only=False, test_only=False)










# # Validation function (evaluates model on validation set)
# def validate(val_dataloader, model, device):
#     model.eval()  # Set model to evaluation mode
#     total_loss = 0
#     correct = 0
#     total = 0
#     criterion = nn.CrossEntropyLoss()

#     with torch.no_grad():
#         bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

#         for i, (imgs, gps) in bar:
#             imgs = imgs.to(device)
#             gps = gps.to(device)

#             gps_queue = model.get_gps_queue()

#             # Forward pass: compute similarity between images and GPS embeddings
#             gps_all = torch.cat([gps, gps_queue], dim=0)
#             logits_img_gps = model(imgs, gps_all)

#             # Compute the validation loss
#             val_loss = criterion(logits_img_gps, torch.arange(imgs.size(0)).long().to(device))
#             total_loss += val_loss.item()

#             # Calculate accuracy (for simplicity, we assume top-1 accuracy)
#             preds = torch.argmax(logits_img_gps, dim=1)
#             correct += (preds == torch.arange(imgs.size(0)).to(device)).sum().item()
#             total += imgs.size(0)

#         avg_loss = total_loss / len(val_dataloader)
#         accuracy = correct / total

#         print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
#     return avg_loss, accuracy