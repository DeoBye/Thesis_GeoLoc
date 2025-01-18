import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geopy.distance import geodesic as GD
import pickle
from sklearn.metrics.pairwise import haversine_distances, cosine_distances

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
            num_far = int(len(distances) * 0.5)             # ratio to split

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
        self.first_train = True
    
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
        negatives = torch.unique(negatives, dim=0)

        max_attempts = 5  # 最大采样尝试次数
        attempt = 0

        while negatives.size(0) < (per_neg_size*gps.shape[0]) and attempt < max_attempts:
            additional_indices = torch.randint(
                0, gps_gallery.shape[0],
                (self.model.queue_size - negatives.size(0),))
            additional_negatives = gps_gallery[additional_indices].to(self.device)
            
            # 合并并去重
            negatives = torch.cat((negatives, additional_negatives), dim=0)
            negatives = torch.unique(negatives, dim=0)
            attempt += 1

        negatives = negatives[:self.model.queue_size]
        indices = torch.randperm(negatives.size(0))
        negatives = negatives[indices]
        # negatives = torch.tensor(negatives)   
        return negatives
    
    def get_negatives2(self, gps):
        #### 和get_negative 一样选取negatives  但选取的少，但是用negatives替换掉queue然后返回
        
        gps_queue = self.model.get_gps_queue()
        # self.model.dequeue_and_euqueue(gps)
        
        gps_gallery = self.model.gps_gallery
        
        per_neg_size = self.model.queue_size // (gps.shape[0] * 4)
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
        negatives = torch.unique(negatives, dim=0)

        while negatives.size(0) < self.model.queue_size:
            additional_indices = torch.randint(
                0, gps_gallery.size(0),
                (self.model.queue_size - negatives.shape[0],),
                device=self.device
            )
            additional_negatives = gps_gallery[additional_indices]
            negatives = torch.cat((negatives, additional_negatives), dim=0)

        negatives = negatives[:self.model.queue_size]
        
        
        indices = torch.randperm(negatives.size(0))
        negatives = negatives[indices]
        # negatives = torch.tensor(negatives)  
        selection = torch.tensor(np.random.choice(gps_queue.shape[0], size=negatives.shape[0], replace=False))
        gps_queue[selection] = negatives
         
        return gps_queue
    
    def get_negatives3(self, gps):
        #### 每次选取一部分hard negatives（batch_size=128时， 每个点选16个近的, 不要远的）替换掉gps_queue的一部分
        ### 现在改成最近的16个+随机选的16个远的。替换掉queue的一部分
        if self.first_train == True:
            self.first_train = False
            return self.model.get_gps_queue()
        else:
            queue = self.model.get_gps_queue().to(self.device)
            gps_gallery = self.model.gps_gallery
            per_neg_size = self.model.queue_size // (gps.shape[0])        ####
            negatives = []
            
            strategy = "hard_negative"
            for g in gps:
                indices = torch.tensor(np.random.choice(gps_gallery.shape[0], per_neg_size * 5, replace=False))     #####
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
                    near_samples = near_neighbor[: num_near_samples]
                    selected_indices = np.concatenate([near_samples, far_samples])
                    # selected_indices = near_samples                           ######


                select_negs = gps_pool[selected_indices]
                negatives.append(select_negs)

            negatives = torch.stack([neg.clone().detach().to(torch.float32) for neg in negatives])
            negatives = negatives.reshape(-1, negatives.shape[-1])
            negatives = torch.unique(negatives, dim=0)

            indices = torch.randperm(negatives.size(0))
            queue[indices] = negatives
            # negatives = torch.tensor(negatives)   
            return queue
    
    def forward(self, imgs, gps, batch_size):
        gps_queue = self.get_negatives3(gps).to(self.device)               #####
        noise_std = 0.01  
        # gps_queue = gps_queue + torch.randn_like(gps_queue) * noise_std        #####
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (2500 / 111_320)
        # indices = torch.randperm(gps_queue.size(0)) 
        # gps_queue = gps_queue[indices]
        
        gps_all = torch.cat([gps, gps_queue], dim=0).to(self.device)
        # self.model.dequeue_and_euqueue(gps)
        logits_img_gps = self.model(imgs, gps_all)
        targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_img_gps, targets_img_gps)
        self.model.dequeue_and_enqueue(gps)  
        return loss
    
class SemiPenaltyLoss(nn.Module):
    def __init__(self, model, device, **kwargs):
        super(SemiPenaltyLoss, self).__init__() 
        self.model = model
        self.device = device
        self.queue_initialized = False
        with open('osv5m_fitted.pkl', 'rb') as fit:
            self.semivar = pickle.load(fit)
        
    def get_penalty(self, imgs, gps, gps_queue, img_queue):
        gps_rad = gps * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        gps_queue_rad = gps_queue * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        
        gc_dist = haversine_distances(gps, gps_queue)
        cosine_dist = cosine_distances(imgs, img_queue)
        
        
    
    def forward(self, imgs, gps, batch_size):
        gps_queue = self.model.get_gps_queue()
        img_queue = self.model.get_img_queue()
        
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        gps_all = torch.cat([gps, gps_queue], dim=0)
        # self.model.dequeue_and_euqueue(gps)
        logits_img_gps = self.model(imgs, gps_all)
        targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_img_gps, targets_img_gps)
        self.model.dequeue_and_enqueue(gps) 
        self.model.img_dequeue_and_enqueue(imgs) 
        
        if int(self.model.img_queue_ptr[0]) == 0:
            self.queue_initialized = True
        return loss
        
 
        
        
class WeightedLogits(nn.Module):
    def __init__(self, model, device, **kwargs):
        super(WeightedLogits, self).__init__() 
        self.model = model
        self.device = device
        
    
    def forward(self, imgs, gps, batch_size):
        gps_queue = self.model.get_gps_queue()
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        gps_all = torch.cat([gps, gps_queue], dim=0)
        # self.model.dequeue_and_euqueue(gps)
        logits_img_gps = self.model(imgs, gps_all)
        
        # pos_logits = torch.diag(logits_img_gps)
        
        neg_mask = torch.ones_like(logits_img_gps, dtype=bool)
        for i in range(batch_size):
            neg_mask[i, i] = False 

        neg_logits = logits_img_gps.masked_select(neg_mask).view(batch_size, -1)

        # Hardness reweighting
        hard_weights = torch.exp(1.0 * neg_logits.detach())
        normalized_weights = hard_weights / hard_weights.mean(dim=-1, keepdim=True) 
        weighted_neg_logits = neg_logits * normalized_weights  # Reweighted negatives
        
        updated_logits_img_gps = logits_img_gps.clone()
        neg_mask_indices = neg_mask.nonzero(as_tuple=True)  # Get negative indices
        batch_indices, neg_indices = neg_mask_indices[0], neg_mask_indices[1]
        updated_logits_img_gps[batch_indices, neg_indices] = weighted_neg_logits.flatten()
        
        
        # logits_img_gps[neg_mask] = weighted_neg_logits
        
        targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(updated_logits_img_gps, targets_img_gps)
        self.model.dequeue_and_enqueue(gps)  
        return loss
    
    
class WeightedLogits2(nn.Module):
    def __init__(self, model, device, **kwargs):
        super(WeightedLogits2, self).__init__() 
        self.model = model
        self.device = device
        self.tau_plus = 0.1
        
    
    def forward(self, imgs, gps, batch_size):
        
        gps_queue = self.model.get_gps_queue()
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        gps_all = torch.cat([gps, gps_queue], dim=0)
        # self.model.dequeue_and_euqueue(gps)
        logits_img_gps = torch.exp(self.model(imgs, gps_all))
        
        pos_logits = torch.diag(logits_img_gps)
        
        neg_mask = torch.ones_like(logits_img_gps, dtype=bool)
        for i in range(batch_size):
            neg_mask[i, i] = False 

        neg_logits = logits_img_gps.masked_select(neg_mask).view(batch_size, -1)

        imp = (1.0 * neg_logits.log()).exp()                       ## beta
        reweight_neg = (imp * neg_logits).sum(dim=-1) / imp.mean(dim=-1)
        N = neg_logits.size(1)
        Ng = (-self.tau_plus * N * pos_logits + reweight_neg) / (1 - self.tau_plus)
        temperature = self.model.logit_scale.exp().detach()
        Ng = torch.clamp(Ng, min=N * torch.exp(-temperature))
        
        loss = (-torch.log(pos_logits / (pos_logits + Ng))).mean()
        # hard_weights = torch.exp(1.0 * neg_logits.detach())
        # normalized_weights = hard_weights / hard_weights.sum(dim=1, keepdim=True) 
        # weighted_neg_logits = neg_logits * normalized_weights  # Reweighted negatives
        # logits_img_gps[neg_mask] = weighted_neg_logits
        
        # targets_img_gps = torch.Tensor([i for i in range(batch_size)]).long().to(self.device)
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(logits_img_gps, targets_img_gps)
        self.model.dequeue_and_enqueue(gps)  
        return loss