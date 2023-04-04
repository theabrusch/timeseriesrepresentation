import torch
import torch.nn.functional as F
import numpy as np

class TS2VecLoss(torch.nn.Module):
    def __init__(self, alpha, temporal_unit) -> None:
        super().__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.maxpool = torch.nn.MaxPool1d(2)
    
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
            z1, z2 = self.maxpool(z1), self.maxpool(z2)
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

    def __init__(self, device, tau, use_cosine_similarity = True):
        super(ContrastiveLoss, self).__init__()
        self.temperature = tau
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, reduce = True):
        if zis is None or zjs is None:
            return torch.tensor(0.).to(self.device)
        batch_size = zis.shape[0]
        if len(zis.shape) > 2:
            # flatten
            zis = zis.view(batch_size, -1)
            zjs = zjs.view(batch_size, -1)
        
        
        representations = torch.cat([zjs, zis], dim=0)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size=batch_size).type(torch.bool)

        similarity_matrix = self.similarity_function(representations, representations)
        if len(similarity_matrix.shape) > 2:
            similarity_matrix = similarity_matrix.mean(dim = 2)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * batch_size, 1),torch.zeros(2 * batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * batch_size) + epsilon * (1/batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss


def compute_weights(targets):
    _, count = np.unique(targets, return_counts=True)
    weights = 1 / count
    weights = weights / weights.sum()
    return torch.tensor(weights).float()
