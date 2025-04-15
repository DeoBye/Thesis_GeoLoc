import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geopy.distance import geodesic as GD
import pickle
from sklearn.metrics.pairwise import haversine_distances, cosine_distances
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import random
import os

class Clip_Inspired:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def __call__(self, inputs):
        gps = inputs['gps'].to(self.device)
        aug1 = inputs['aug1'].to(self.device)
        gps_queue = self.model.get_gps_queue()
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        gps_noise = gps + torch.randn_like(gps) * (150 / 111_320)
        gps_all = torch.cat([gps_noise, gps_queue], dim=0)
        logits_img_gps = self.model(aug1, gps_all)
        self.model.dequeue_and_enqueue(gps)
        ret_dict = {"logits": logits_img_gps}
        return ret_dict
    
    
    
class SSL:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def __call__(self, inputs):
        gps = inputs['gps'].to(self.device)
        aug1 = inputs['aug1'].to(self.device)
        aug2 = inputs['aug2'].to(self.device)
        gps_queue = self.model.get_gps_queue()
        gps_queue = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        gps_noise = gps + torch.randn_like(gps) * (150 / 111_320)
        gps_all = torch.cat([gps_noise, gps_queue], dim=0)
        aug1_features = self.model.image_encoder(aug1)
        aug2_features = self.model.image_encoder(aug2)
        z1 = self.model.projector(aug1_features)
        z2 = self.model.projector(aug2_features)
        p1 = self.model.predictor(z1)
        p2 = self.model.predictor(z2)
        location_features = self.model.location_encoder(gps_all)
        logits_aug1_gps = self.model(aug1, gps_all)
        logits_aug2_gps = self.model(aug2, gps_all)
        self.model.dequeue_and_enqueue(gps)
        ret_dict = {"logits1": logits_aug1_gps, 
                    "logits2": logits_aug2_gps, 
                    "z1": z1,
                    "z2": z2,
                    "p1": p1,
                    "p2": p2}
        return ret_dict
            
            
class Triplet:
    def __init__(self, model, device):
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
    
    
    def __call__(self, inputs):
        imgs = inputs['aug1'].to(self.device)
        gps = inputs['gps'].to(self.device)
        batch_size = len(gps)
        
        anchors = self.model.image_encoder(imgs)
        positives = self.model.location_encoder(gps)
        
        indices = torch.tensor(np.random.choice(self.model.gps_gallery.shape[0], batch_size*4, replace=False))
        gps_pool = self.model.gps_gallery[indices].to(self.device)
        
        negatives = self.get_negatives(gps, gps_pool).to(self.device)
        
        negatives = self.model.location_encoder(negatives)
        
        dist_fn = 'cos'                                  #####
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        ret_dict = {'anchors': anchors,
                    'positives': positives,
                    'negatives': negatives}
        
        if dist_fn == 'Euc':
            loss_fn = nn.TripletMarginLoss()
            loss = loss_fn(anchors, positives, negatives)
        if dist_fn == 'cos':
            cosine_similarity = nn.CosineSimilarity(dim=1)
            loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - cosine_similarity(x, y),
                                                    margin=1.0)
            loss = loss_fn(anchors, positives, negatives)
            
        return ret_dict
        
class EntropywithDis:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.first_train = True
        gps_radians = np.radians(self.model.gps_gallery.cpu().numpy())  # Convert to radians
        self.tree = BallTree(gps_radians, metric='haversine')
        
    def visualize_negatives(self, gps, negatives, index):
        """Visualize true GPS and selected negatives."""
        gps = gps.cpu().numpy()
        negatives = negatives.cpu().numpy()
        
        save_dir = '/root/geo-clip/experiments/record/neg5_5000/'
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.scatter(negatives[:, 1], negatives[:, 0], c='blue', s=5, alpha=0.6, label='Negative Samples')
        plt.scatter(gps[:, 1], gps[:, 0], c='red', s=1, label='True GPS')
        
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.title("True GPS vs. Selected Negative Samples")
        plt.savefig(f"{save_dir}_{index}.png")
        np.save(f"{save_dir}_{index}.npy", gps)
        np.save(f"{save_dir}_{index}.npy", negatives)
    
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

        max_attempts = 5  # max re-sample steps
        attempt = 0

        while negatives.size(0) < (per_neg_size*gps.shape[0]) and attempt < max_attempts:
            additional_indices = torch.randint(
                0, gps_gallery.shape[0],
                (self.model.queue_size - negatives.size(0),))
            additional_negatives = gps_gallery[additional_indices].to(self.device)
            
            # merge and delete duplicate
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
        
        
    def filter_out_true_gps(self, all_neg_indices, gps):
        """Filter out candidates that match true GPS coordinates."""
        true_gps_set = set([tuple(coord) for coord in gps.cpu().numpy()])
        filtered_indices = [
            idx for idx in all_neg_indices 
            if tuple(self.model.gps_gallery[idx].tolist()) not in true_gps_set
        ]
        return filtered_indices
        
    def get_negatives4(self, gps):

        # k = gps.shape[0] * 5  # Query 10x the batch size for diverse neighbors
        gps_radians = np.radians(gps.cpu().numpy())
        
        # Query BallTree for k nearest neighbors for each GPS point
        dist, indices = self.tree.query(gps_radians, k=9)
        all_neg_indices = indices.reshape(-1)  # Flatten the indices

        # Filter out any negative that matches the true GPS coordinates
        filtered_indices = self.filter_out_true_gps(all_neg_indices, gps)
        
        filtered_indices = np.unique(filtered_indices)
        
        # Randomly sample from the filtered set
        num_negatives = self.model.queue_size
        selected_indices = np.random.choice(filtered_indices, size=num_negatives, replace=False)
        
        negatives = torch.tensor(self.model.gps_gallery[selected_indices], dtype=torch.float32).to(self.device)
        return negatives
    
    def get_negatives5(self, gps):

        # k = gps.shape[0] * 5  # Query 10x the batch size for diverse neighbors
    
        gps_radians = np.radians(gps.cpu().numpy())

    # First query: k nearest neighbors (near neighbors)
        k = 6
        dist1, indices1 = self.tree.query(gps_radians, k=k)
        all_near_indices = indices1.reshape(-1)

        # Second query: 2*k nearest neighbors (near + farther neighbors)
        dist2, indices2 = self.tree.query(gps_radians, k=4*k)
        all_far_indices = indices2.reshape(-1)

        # Exclude near neighbors from the far neighbors
        far_only_indices = np.setdiff1d(all_far_indices, all_near_indices)

        # Randomly sample from both sets
        num_negatives = self.model.queue_size
        num_near_samples = int(num_negatives * 0.5)
        num_far_samples = num_negatives - num_near_samples

        all_near_indices = np.unique(self.filter_out_true_gps(all_near_indices, gps))
        all_far_indices = np.unique(self.filter_out_true_gps(all_far_indices, gps))
        
        
        sampled_near = np.random.choice(all_near_indices, size=min(num_near_samples, len(all_near_indices)), replace=False)
        sampled_far = np.random.choice(far_only_indices, size=min(num_far_samples, len(far_only_indices)), replace=False)

        # Concatenate near and far samples
        selected_indices = np.concatenate([sampled_near, sampled_far])
        
        negatives = torch.tensor(self.model.gps_gallery[selected_indices], dtype=torch.float32).to(self.device)
        return negatives
    
    def __call__(self, inputs):
        imgs = inputs['aug1'].to(self.device)
        gps = inputs['gps'].to(self.device)
        batch_size = len(gps)
        gps_queue = self.get_negatives4(gps).to(self.device)               #####
        noise_std = 0.01  
        # gps_queue = gps_queue + torch.randn_like(gps_queue) * noise_std        #####
        gps_noise = gps + torch.randn_like(gps) * (150 / 111_320)
        gps_queue_noise = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        # indices = torch.randperm(gps_queue.size(0)) 
        # gps_queue = gps_queue[indices]
        
        gps_all = torch.cat([gps_noise, gps_queue_noise], dim=0).to(self.device)
        logits_img_gps = self.model(imgs, gps_all)
        self.model.dequeue_and_enqueue(gps)  
        
        if random.random() < 0.1:
            self.visualize_negatives(gps, gps_queue, index=random.randint(1, 10000))
            
        ret_dict = {'logits': logits_img_gps}
        return ret_dict
    
class SemivarioPenalty:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.queue_initialized = False
        with open('osv5m_fitted.pkl', 'rb') as fit:
            self.semivar = pickle.load(fit)
            self.rho_thres = 1e-2
            t = np.arange(0, np.pi, 0.01)
            res = self.semivar.variogram(t)
            der = (res[1:] - res[:-1]) / 0.01
            self.rho = np.where(der < self.rho_thres)[0][0] * 0.01
        
    def get_penalty(self, logits, imgs_emb, gps, gps_all, img_all):
        ## 直接用差值来做logits
        gps_rad = gps.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        gps_all_rad = gps_all.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        
        gc_dist = haversine_distances(gps_rad, gps_all_rad)
        cosine_dist = cosine_distances(imgs_emb.cpu().numpy(), img_all.cpu().numpy())
        
        pred_dist = self.semivar.variogram(gc_dist)
        
        penalty = np.abs(pred_dist - cosine_dist)
        penalty = torch.clamp(torch.from_numpy(penalty).to(self.device).to(logits.dtype), min=logits.min(), max = logits.max())
        
        batch_size = gps.shape[0]
        neg_mask = torch.ones_like(logits, dtype=bool)
        for i in range(batch_size):
            neg_mask[i, i] = False 
        
        updated_logits_img_gps = logits.clone()
        neg_mask_indices = neg_mask.nonzero(as_tuple=True)  # Get negative indices
        batch_indices, neg_indices = neg_mask_indices[0], neg_mask_indices[1]
        updated_logits_img_gps[batch_indices, neg_indices] = penalty[batch_indices, neg_indices]
        
        return updated_logits_img_gps
    
    
    def get_penalty2(self, logits, imgs_emb, gps, gps_all, img_all):
        ## 差值作为beta来 reweight
        gps_rad = gps.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        gps_all_rad = gps_all.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        
        gc_dist = haversine_distances(gps_rad, gps_all_rad)
        cosine_dist = cosine_distances(imgs_emb.cpu().numpy(), img_all.cpu().numpy())
        
        pred_dist = self.semivar.variogram(gc_dist)
        
        penalty = np.abs(pred_dist - cosine_dist)
        penalty = torch.clamp(torch.from_numpy(penalty).to(self.device).to(logits.dtype), min=0.2, max = 0.5)        
        
        batch_size = gps.shape[0]
        neg_mask = torch.ones_like(logits, dtype=bool)
        for i in range(batch_size):
            neg_mask[i, i] = False 
            
        neg_logits = logits.masked_select(neg_mask).view(batch_size, -1)
        neg_penalty = penalty.masked_select(neg_mask).view(batch_size, -1)
        
        hard_weights = torch.exp(neg_penalty * neg_logits.detach())
        normalized_weights = hard_weights / hard_weights.mean(dim=-1, keepdim=True) 
        weighted_neg_logits = neg_logits * normalized_weights  # Reweighted negatives
        
        updated_logits_img_gps = logits.clone()
        neg_mask_indices = neg_mask.nonzero(as_tuple=True)  # Get negative indices
        batch_indices, neg_indices = neg_mask_indices[0], neg_mask_indices[1]
        updated_logits_img_gps[batch_indices, neg_indices] = weighted_neg_logits.flatten()
        
        return updated_logits_img_gps

    def get_penalty3(self, logits, imgs_emb, gps, gps_all, img_all):
        ## 直接用差值来做logits
        gps_rad = gps.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        gps_all_rad = gps_all.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        
        gc_dist = haversine_distances(gps_rad, gps_all_rad)
        cosine_dist = cosine_distances(imgs_emb.cpu().numpy(), img_all.cpu().numpy())
        
        pred_dist = self.semivar.variogram(gc_dist)
        
        deriv = np.abs((pred_dist[1:] - pred_dist[:-1]))
        rho_thres = 1e-6
        rho_idx = np.where(deriv < rho_thres)
        
        penalty = np.maximum(0, cosine_dist - pred_dist)
        penalty[rho_idx] = 0   
        
        penalty = torch.clamp(torch.from_numpy(penalty).to(self.device).to(logits.dtype), min=logits.min(), max = logits.max())
        
        batch_size = gps.shape[0]
        neg_mask = torch.ones_like(logits, dtype=bool)
        for i in range(batch_size):
            neg_mask[i, i] = False 
        
        updated_logits_img_gps = logits.clone()
        neg_mask_indices = neg_mask.nonzero(as_tuple=True)  # Get negative indices
        batch_indices, neg_indices = neg_mask_indices[0], neg_mask_indices[1]
        updated_logits_img_gps[batch_indices, neg_indices] = penalty[batch_indices, neg_indices]
        
        return updated_logits_img_gps
    
    
    def get_penalty4(self, logits, imgs_emb, gps, gps_all, img_all):
        ## 差值作为beta来 reweight
        gps_rad = gps.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        gps_all_rad = gps_all.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        
        gc_dist = haversine_distances(gps_rad, gps_all_rad)
        cosine_dist = cosine_distances(imgs_emb.cpu().numpy(), img_all.cpu().numpy())
        
        pred_dist = self.semivar.variogram(gc_dist)
        rho_thres = 1e-6
        deriv = np.abs((pred_dist[1:] - pred_dist[:-1]))
        rho_idx = np.where(deriv < rho_thres)
        
        penalty = np.maximum(0, cosine_dist - pred_dist)
        penalty[rho_idx] = 0   
        
        penalty = torch.from_numpy(penalty.reshape(cosine_dist.shape)).to(self.device).to(logits.dtype)
               
        
        batch_size = gps.shape[0]
        neg_mask = torch.ones_like(logits, dtype=bool)
        for i in range(batch_size):
            neg_mask[i, i] = False 
            
        neg_logits = logits.masked_select(neg_mask).view(batch_size, -1)
        neg_penalty = penalty.masked_select(neg_mask).view(batch_size, -1)
        
        hard_weights = torch.exp(neg_penalty * neg_logits.detach())
        normalized_weights = hard_weights / hard_weights.mean(dim=-1, keepdim=True) 
        weighted_neg_logits = neg_logits * normalized_weights  # Reweighted negatives
        
        updated_logits_img_gps = logits.clone()
        neg_mask_indices = neg_mask.nonzero(as_tuple=True)  # Get negative indices
        batch_indices, neg_indices = neg_mask_indices[0], neg_mask_indices[1]
        updated_logits_img_gps[batch_indices, neg_indices] = weighted_neg_logits.flatten()
        
        return updated_logits_img_gps   
    
    def get_penalty5(self, logits, imgs_emb, gps, gps_all, img_all):
        ## 差值作为beta来 reweight
        gps_rad = gps.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        gps_all_rad = gps_all.cpu().numpy() * np.array([np.pi/2, np.pi]) / np.array([90, 180])
        
        gc_dist = haversine_distances(gps_rad, gps_all_rad)
        cosine_dist = cosine_distances(imgs_emb.cpu().numpy(), img_all.cpu().numpy())
        
        gc_flat = gc_dist.flatten()
        cosine_flat = cosine_dist.flatten()
        
        semi_dist = self.semivar.variogram(gc_flat)
        
        # sort_idx = np.argsort(gc_flat)
        # gc_sorted = gc_flat[sort_idx]
        # pred_sorted = pred_dist[sort_idx]
        # deriv = np.abs(pred_sorted[1:] - pred_sorted[:-1])
        # flat_idx_sorted = np.where(deriv < rho_thres)[0]       
        # flat_idx = sort_idx[flat_idx_sorted]
        
        
        penalty = np.maximum(0, cosine_flat - semi_dist)
        penalty[gc_flat > self.rho] = 0     ## penalty dist > rho = 0
        penalty = penalty.reshape(gc_dist.shape)
         
        
        penalty = torch.from_numpy(penalty.reshape(cosine_dist.shape)).to(self.device).to(logits.dtype)
        batch_size = gps.shape[0]
        neg_mask = torch.ones_like(logits, dtype=bool)
        for i in range(batch_size):
            neg_mask[i, i] = False 
            
        neg_logits = logits.masked_select(neg_mask).view(batch_size, -1)
        neg_penalty = penalty.masked_select(neg_mask).view(batch_size, -1)
        
        weight = torch.exp(neg_penalty)             ## penalty ranges from 0 - 0.8              exp(penalty / 2)  ranges 1 - 1.5
                                                    ## stabilize in the range of 0 - 0.4 
        
        weighted_neg_logits = neg_logits * weight  # Reweighted neg logits
        
        updated_logits_img_gps = logits.clone()
        neg_mask_indices = neg_mask.nonzero(as_tuple=True)  # Get negative indices
        batch_indices, neg_indices = neg_mask_indices[0], neg_mask_indices[1]
        updated_logits_img_gps[batch_indices, neg_indices] = weighted_neg_logits.flatten()
        
        return updated_logits_img_gps        
        
    
    def __call__(self, inputs):
        gps_queue = self.model.get_gps_queue()
        img_queue = self.model.get_img_queue() 
        gps = inputs['gps'].to(self.device)
        imgs = inputs['aug1'].to(self.device)
        batch_size = len(gps)
        gps_noise = gps + torch.randn_like(gps) * (150 / 111_320)
        gps_queue_noise = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)            
        gps_all = torch.cat([gps, gps_queue], dim=0)
        gps_all_noise = torch.cat([gps_noise, gps_queue_noise], dim=0)
        
        # self.model.dequeue_and_euqueue(gps)
        logits_img_gps = self.model(imgs, gps_all_noise)
        imgs_emb = self.model.image_encoder.CLIP.get_image_features(pixel_values=imgs)
        
        if self.queue_initialized:
            img_all = torch.cat([imgs_emb, img_queue], dim=0)
            logits_img_gps = self.get_penalty5(logits_img_gps, imgs_emb, gps, gps_all, img_all)
            
        self.model.dequeue_and_enqueue(gps) 
        self.model.img_dequeue_and_enqueue(imgs_emb) 
        
        if int(self.model.img_queue_ptr[0]) == 0:
            self.queue_initialized = True
            
        ret_dict = {'logits': logits_img_gps}
        return ret_dict
        
 
        
        
class WeightedLogits(nn.Module):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    
    def __call__(self, inputs):
        gps = inputs['gps'].to(self.device)
        imgs = inputs['aug1'].to(self.device)
        batch_size = len(gps)        
        gps_queue = self.model.get_gps_queue()
        gps_noise = gps + torch.randn_like(gps) * (150 / 111_320)
        gps_queue_noise = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        gps_all = torch.cat([gps_noise, gps_queue_noise], dim=0)
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
        ret_dict = {'logits': updated_logits_img_gps}
        self.model.dequeue_and_enqueue(gps)  
        return ret_dict
    
    
class WeightedLogits2:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.tau_plus = 0.1
        
    
    def __call__(self, inputs):
        gps = inputs['gps'].to(self.device)
        imgs = inputs['aug1'].to(self.device)
        batch_size = len(gps)
        
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
        ret_dict = {'reweight_loss': loss}
        return ret_dict
    
class WeightedwithNeg:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.first_train = True
        gps_radians = np.radians(self.model.gps_gallery.cpu().numpy()) 
        self.tree = BallTree(gps_radians, metric='haversine')

    def filter_out_true_gps(self, all_neg_indices, gps):
        """Filter out candidates that match true GPS coordinates."""
        true_gps_set = set([tuple(coord) for coord in gps.cpu().numpy()])
        filtered_indices = [
            idx for idx in all_neg_indices 
            if tuple(self.model.gps_gallery[idx].tolist()) not in true_gps_set
        ]
        return filtered_indices
    
    def get_neg(self, gps):

        # k = gps.shape[0] * 5  # Query 10x the batch size for diverse neighbors
    
        gps_radians = np.radians(gps.cpu().numpy())

    # First query: k nearest neighbors (near neighbors)
        k = 8
        dist1, indices1 = self.tree.query(gps_radians, k=k)
        all_near_indices = indices1.reshape(-1)

        # Second query: 2*k nearest neighbors (near + farther neighbors)
        dist2, indices2 = self.tree.query(gps_radians, k=2*k)
        all_far_indices = indices2.reshape(-1)

        # Exclude near neighbors from the far neighbors
        far_only_indices = np.setdiff1d(all_far_indices, all_near_indices)

        # Randomly sample from both sets
        num_negatives = self.model.queue_size
        num_near_samples = int(num_negatives * 0.5)
        num_far_samples = num_negatives - num_near_samples

        all_near_indices = np.unique(self.filter_out_true_gps(all_near_indices, gps))
        all_far_indices = np.unique(self.filter_out_true_gps(all_far_indices, gps))
        
        
        sampled_near = np.random.choice(all_near_indices, size=min(num_near_samples, len(all_near_indices)), replace=False)
        sampled_far = np.random.choice(far_only_indices, size=min(num_far_samples, len(far_only_indices)), replace=False)

        # Concatenate near and far samples
        selected_indices = np.concatenate([sampled_near, sampled_far])
        
        negatives = torch.tensor(self.model.gps_gallery[selected_indices], dtype=torch.float32).to(self.device)
        return negatives
    
    def __call__(self, inputs):
        gps = inputs['gps'].to(self.device)
        imgs = inputs['aug1'].to(self.device)
        batch_size = len(gps)
        gps_queue = self.get_neg(gps).to(self.device)               #####
        noise_std = 0.01  
        # gps_queue = gps_queue + torch.randn_like(gps_queue) * noise_std        #####
        gps_noise = gps + torch.randn_like(gps) * (150 / 111_320)
        gps_queue_noise = gps_queue + torch.randn_like(gps_queue) * (1000 / 111_320)
        
        gps_all = torch.cat([gps_noise, gps_queue_noise], dim=0)
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
        
        ret_dict = {'logits': updated_logits_img_gps}
        self.model.dequeue_and_enqueue(gps)  
        return ret_dict
    
