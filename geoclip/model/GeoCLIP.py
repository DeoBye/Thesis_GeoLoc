import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=512, num_layers=3):
        super(projection_MLP, self).__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        # self.bn2 = BN(hidden_dim)

        if self.num_layers == 3:
            self.relu2 = nn.ReLU(inplace=True)
            self.linear3 = nn.Linear(hidden_dim, out_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            # self.bn3 = BN(hidden_dim)

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # b, _ = x.shape
        # layer 1
        x = self.linear1(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        # x.reshape(b, self.hidden_dim)

        # layer 2
        x = self.linear2(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn2(x)


        if self.num_layers == 3:
            x = self.relu2(x)
            # x.reshape(b, self.hidden_dim)
            # layer 3
            x = self.linear3(x)
            # x.reshape(b, self.out_dim, 1)
            x = self.bn3(x)
            # x.reshape(b, self.out_dim)

        return x

class prediction_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=512): # bottleneck structure
        super(prediction_MLP, self).__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers.
        The dimension of h’s input and output (z and p) is d = 2048,
        and h’s hidden layer’s dimension is 512, making h a
        bottleneck structure (ablation in supplement).
        '''
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing.
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        b, _ = x.shape

        # layer 1
        x = self.linear1(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        # x.reshape(b, self.hidden_dim)

        x = self.layer2(x)
        return x

class GeoCLIP(nn.Module):
    def __init__(self, strategy, from_pretrained=True, queue_size=4096):                      ######
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()
        self.projector = projection_MLP(in_dim=512)
        # self.projector = projection_MLP(1024)
        self.predictor = prediction_MLP(512)
        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv"))
        self._initialize_gps_queue(queue_size)
        self._initialize_img_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            print('from pretrain')
            self._load_weights()

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.projector.to(device)
        self.predictor.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth"))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth"))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth"))
        # self.image_encoder.mlp.load_state_dict(torch.load("/root/geo-clip/experiments/EntropyWithDis/epoch_10/image_encoder_mlp_weights.pth", map_location="cpu"))
        #3self.location_encoder.load_state_dict(torch.load("/root/geo-clip/experiments/EntropyWithDis/epoch_10/location_encoder_weights.pth", map_location="cpu"))
        # self.logit_scale = nn.Parameter(torch.load("/root/geo-clip/experiments/EntropyWithDis/epoch_10/logit_scale_weights.pth",map_location="cpu"))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def _initialize_img_queue(self, queue_size):
        self.register_buffer("img_queue", torch.randn(768, self.queue_size))
        self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
        self.register_buffer("img_queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def img_dequeue_and_enqueue(self, imgs):
        img_batch_size = imgs.shape[0]
        img_ptr = int(self.img_queue_ptr)
        
        assert self.queue_size % img_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {img_batch_size}"
        self.img_queue[:, img_ptr:img_ptr + img_batch_size] = imgs.t()
        img_ptr = (img_ptr + img_batch_size) % self.queue_size
        self.img_queue_ptr[0] = img_ptr


    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
    
    def get_img_queue(self):
        return self.img_queue.t()        
        
    # def compute_logits(self, image_features, location_features, with_scale=True):
    #             # Normalize features
    #     image_features = F.normalize(image_features, dim=1)
    #     location_features = F.normalize(location_features, dim=1)
        
    #     # Cosine similarity (Image Features & Location Features)
    #     if with_scale:
    #         logit_scale = self.logit_scale.exp()
    #         logits_per_image = logit_scale * (image_features @ location_features.t())
    #     else:
    #         logits_per_image = image_features @ location_features.t()
            
    #     return logits_per_image  
                 
    def forward(self, image, location):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """

        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Cosine similarity (Image Features & Location Features)
        triple = False
        if not triple:
            logits_per_image = logit_scale * (image_features @ location_features.t())
        else:
            logits_per_image = image_features @ location_features.t()
        
        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob