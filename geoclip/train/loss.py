import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def D(p, z):
    # [N, E]
    z = z.detach() # stop gradient
    p = p / p.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    # [N E] [N E] -> [N]
    return (p * z).sum(dim=1).mean() # dot product & batch coeff normalization

def D_minimize(p, z):  # ..., X, size; ..., Y, size; choose the minimize one
    z = z.detach()
    p = p / p.norm(dim=-1, keepdim=True)
    z = (z / z.norm(dim=-1, keepdim=True)).permute(0, 2, 1)
    sim = torch.bmm(p, z)
    return sim.max(dim=-1)[0].mean(dim=-1).mean()


class SimsiamLoss(nn.Module):
    def __init__(self, symmetry=True):
        super(SimsiamLoss, self).__init__()
        self.symmetry = symmetry

    def forward(self, p1, z1, p2, z2, minimize_loss=False,):
        if self.symmetry:
            if minimize_loss:
                D1 = D_minimize(p1, z2)
                D2 = D_minimize(p2, z1)
                # import ipdb
                # ipdb.set_trace()
                return -0.5 * (D1.mean() + D2.mean())
            else:
                D1 = D(p1, z2)
                D2 = D(p2, z1)
                return -0.5 * (D(p1, z2)  + D(p2, z1) )
            
            
def get_loss(outputs, strategy_name):
    loss_dict = {}

    if strategy_name == "ssl":
        simsiam_criterion = SimsiamLoss()
        targets = torch.arange(len(outputs["logits1"]), device=outputs["logits1"].device)

        clip_loss1 = F.cross_entropy(outputs["logits1"], targets)
        clip_loss2 = F.cross_entropy(outputs["logits2"], targets)
        clip_loss = 0.5 * (clip_loss1 + clip_loss2)

        simsiam_loss = simsiam_criterion(outputs["p1"], outputs["z1"], outputs["p2"], outputs["z2"])

        loss_dict["clip_loss"] = clip_loss
        loss_dict["simsiam_loss"] = simsiam_loss
        loss_dict["loss"] = 0.9 * clip_loss + 0.1 * simsiam_loss
        
    elif strategy_name == "triplet":
        cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        triplet_loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - cosine_similarity(x, y),
            margin=1.0
        )
        triplet_loss = triplet_loss_fn(
            outputs["anchors"], outputs["positives"], outputs["negatives"]
        )
        loss_dict["loss"] = triplet_loss

    elif strategy_name == "weighted2":
        loss_dict["loss"] = outputs["reweight_loss"]

    elif strategy_name in ["clip", "entropy", "semi", "weighted", "weighted_neg"]:
        targets = torch.arange(len(outputs["logits"]), device=outputs['logits'].device)
        loss = F.cross_entropy(outputs["logits"], targets)
        loss_dict["loss"] = loss

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return loss_dict