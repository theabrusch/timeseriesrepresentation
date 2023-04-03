import torch
import torch.nn.functional as F
import numpy as np

class TS2VecLoss(torch.nn.Module):
    def __init__(self, alpha, temporal_unit) -> None:
        super().__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit
    
    def dual_loss(self, z1, z2, d):
        # z1, z2 : B x C x T
        dual_loss = torch.tensor(0.).to(z1.device)
        inst_loss = torch.tensor(0.)

        inst_loss = self.contrastive_loss(z1, z2)
        

        if self.alpha > 0:
            # compute instance loss
            dual_loss += self.alpha*inst_loss
        if self.alpha < 1 and d >= self.temporal_unit:
            # compute temporal loss
            temp_loss = self.contrastive_loss(z1.transpose(0,2), z2.transpose(0,2))
            dual_loss += (1-self.alpha)*temp_loss
        else:
            temp_loss = torch.tensor(0.).to(z1.device)

        return dual_loss, inst_loss.detach().cpu(), temp_loss.detach().cpu()

    def forward(self, z1, z2):
        # z1, z2 : B x C x T
        loss, inst_loss, temp_loss = self.dual_loss(z1, z2, d=0)
        d = 1
        while z1.shape[-1] > 1:
            z1, z2 = F.max_pool1d(z1, 2), F.max_pool1d(z2, 2)
            out = self.dual_loss(z1, z2, d)
            loss += out[0]
            inst_loss += out[1]
            temp_loss += out[2]
            d+=1

        return loss/d, inst_loss/d, temp_loss/d

    def contrastive_loss(self, z1, z2):
        '''
        The contrastive loss is computed across the first dimension.
        '''
        # z1, z2 : B x C x T
        B = z1.shape[0]
        z = torch.cat([z1,z2], dim = 0) # 2B x C x T
        z = z.permute((2,0,1)) # T x 2B x C
        sim = torch.matmul(z, z.transpose(1,2)) # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:] 
        logits = -F.log_softmax(logits, dim=-1) 
        
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2 
        return loss

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, tau) -> None:
        super().__init__()
        self.temperature = tau
    def forward(self, z1, z2):
        B = z1.shape[0]
        z = torch.cat([z1,z2], dim = 0) # 2B x C
        sim = torch.matmul(z, z.transpose(0,1)) # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, 1:] 
        logits = -F.log_softmax(logits, dim=-1) 
        logits /= self.temperature
        
        i = torch.arange(B, device=z1.device)
        loss = (logits[i, B + i - 1].mean() + logits[B + i, i].mean()) / 2 
        return loss



def compute_weights(targets):
    _, count = np.unique(targets, return_counts=True)
    weights = 1 / count
    weights = weights / weights.sum()
    return torch.tensor(weights).float()
