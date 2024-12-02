import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, net, image_features, random_text_features, false_text_features):
        """Contrastive loss function with false text features

        Args:
            net (_type_): _description_
            image_features (_type_): _description_
            random_text_features (_type_): _description_
            false_text_features (_type_): _description_
        """
        bs, atn, ndim = image_features.shape 
        false_text_num = false_text_features.shape[0] // bs
        logit_scale = net.logit_scale.exp()
        random_text_features = random_text_features.tensor
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        random_text_features = random_text_features / random_text_features.norm(dim=-1, keepdim=True)
        false_text_features = false_text_features / false_text_features.norm(dim=-1,keepdim = True)
        logits_per_image = torch.zeros((bs, atn)).cuda().to(image_features.device)
        for b in range(bs):
            logits_per_image[b] = logit_scale*torch.sum(image_features[b].unsqueeze(0)*(torch.cat((random_text_features,false_text_features[b*false_text_num:(b+1)*false_text_num]),dim=0)),dim=-1)
        
        logits_per_text = logits_per_image[:,:bs].t()
        
        labels = torch.arange(bs).long().to(logits_per_image.device)
        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2
        losses = []
        losses_name = []
        losses.append(total_loss)
        losses_name.append("contrastive loss")

        return losses, losses_name
