import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geopy.distance import geodesic as GD

class CrossEntropyLoss(nn.Module):
    def __init__(self, model, device, **kwargs):
        super(CrossEntropyLoss, self).__init__() 
        self.model = model
        self.device = device
    
    def forward(self, imgs, gps, batch_size):
        gps_queue = self.model.get_gps_queue()
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        gps_all = torch.cat([gps, gps_queue], dim=0)
        # self.model.dequeue_and_euqueue(gps)
        logits_img_gps = self.model(imgs, gps_all)
        targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_img_gps, targets_img_gps)
        self.model.dequeue_and_enqueue(gps)  
        return loss
        
    
class TripletLoss(nn.Module):
    def __init__(self, model, device, **kwargs):
        super(TripletLoss, self).__init__() 
        self.model = model
        self.device = device
        
    def get_negatives(self, gps, gps_pool):
        negatives = []
        strategy = "hard_negative"
        for g in gps:
            distances = [GD(g, gps_point) for gps_point in gps_pool]
            
            sorted = np.argsort(distances)
            num_far = int(len(distances) * 0.5)            

            near_neighbor = sorted[: len(distances)-num_far]
            far_neighbor = sorted[len(distances)-num_far :]
            
            if strategy == "max_distance":
                max_index = far_neighbor[-1]
                negatives.append(gps_pool[max_index])

            elif strategy == "hard_negative":
                if np.random.rand() < 0.5:
                    # Select from far neighbors
                    selected_index = np.random.choice(far_neighbor)
                else:
                    # Select from near neighbors
                    selected_index = np.random.choice(near_neighbor)

                negatives.append(gps_pool[selected_index])

        negatives = torch.stack([torch.tensor(neg, dtype=torch.float32) for neg in negatives])    
        
        return negatives
    
    
    def forward(self, imgs, gps, batch_size):
        anchors = self.model.image_encoder(imgs)
        positives = self.model.location_encoder(gps)
        
        gps_pool = self.model.gps_gallery
        indices = torch.tensor(np.random.choice(gps_pool.shape[0], batch_size*4, replace=False))
        gps_pool = gps_pool[indices].to(self.device)
        
        negatives = self.get_negatives(gps, gps_pool).to(self.device)
        
        negatives = self.model.location_encoder(negatives)
        dist_fn = 'cos'                                  #####
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        if dist_fn == 'Euc':
            loss_fn = nn.TripletMarginLoss()
            loss = loss_fn(anchors, positives, negatives)
        if dist_fn == 'cos':
            cosine_similarity = nn.CosineSimilarity(dim=1)
            loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - cosine_similarity(x, y),
                                                    margin=1.0)
            loss = loss_fn(anchors, positives, negatives)
            
        return loss
        
class EntropywithDis(nn.Module):
    def __init__(self, model, device, **kwargs):
        super(EntropywithDis, self).__init__() 
        self.model = model
        self.device = device
    
    def get_negatives(self, gps):
        gps_gallery = self.model.gps_gallery
        per_neg_size = self.model.queue_size // gps.shape[0]
        negatives = []
        
        strategy = "hard_negative"
        for g in gps:
            indices = torch.tensor(np.random.choice(gps_gallery.shape[0], per_neg_size * 5, replace=False))
            gps_pool = gps_gallery[indices].to(self.device)    # select 64*5 gps
            distances = [GD(g, gps_point) for gps_point in gps_pool]    # compute the dis between each gps with gps_pool in previous step
            
            sorted = np.argsort(distances)
            num_far = int(len(distances) * 0.7)             # ratio to split

            near_neighbor = sorted[: len(distances)-num_far]
            far_neighbor = sorted[len(distances)-num_far :]
            
            if strategy == "max_distance":
                selected_indices = far_neighbor[-per_neg_size:]

            elif strategy == "hard_negative":
                num_far_samples = int(per_neg_size * 0.5)
                num_near_samples = per_neg_size - num_far_samples
                far_samples = np.random.choice(far_neighbor, size=num_far_samples, replace=False)
                near_samples = np.random.choice(near_neighbor, size=num_near_samples, replace=False)
                selected_indices = np.concatenate([near_samples, far_samples])


            select_negs = gps_pool[selected_indices]
            negatives.append(select_negs)

        negatives = torch.stack([neg.clone().detach().to(torch.float32) for neg in negatives])
        negatives = negatives.reshape(-1, negatives.shape[-1])
        # negatives = torch.tensor(negatives)   
        return negatives
    
    def forward(self, imgs, gps, batch_size):
        gps_queue = self.get_negatives(gps).to(self.device)
        noise_std = 0.01  
        # gps_queue = gps_queue + torch.randn_like(gps_queue) * noise_std        #####
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        indices = torch.randperm(gps_queue.size(0)) 
        gps_queue = gps_queue[indices]
        
        gps_all = torch.cat([gps, gps_queue], dim=0).to(self.device)
        # self.model.dequeue_and_euqueue(gps)
        logits_img_gps = self.model(imgs, gps_all)
        targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_img_gps, targets_img_gps)
        self.model.dequeue_and_enqueue(gps)  
        return loss
        
        
        
        
        