import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 

class conv_block(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=kernel, stride = stride, padding = kernel//2),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layer(x)

def conv1D_out_shape(input_shape, kernel, stride, padding):
    kernel, stride, padding = np.array(kernel), np.array(stride), np.array(padding)
    shape = input_shape
    for kern, stri, padd in zip(kernel, stride, padding):
        shape = int((shape + 2*padd - kern)/stri + 1)
    return shape

class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x

class TFC_encoder(nn.Module):
    def __init__(self, in_channels, input_size, num_classes = None, classify = True):
        super().__init__()
        self.classify = classify

        self.TimeEncoder = nn.Sequential(
            conv_block(channels_in = in_channels, channels_out = 32, kernel = 8, stride = 8, dropout = 0.5),
            conv_block(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = 0.5),
            conv_block(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = 0.5),
            nn.Flatten()
            )
        
        self.FrequencyEncoder = nn.Sequential(
            conv_block(channels_in = in_channels, channels_out = 32, kernel = 8, stride = 8, dropout = 0.5),
            conv_block(channels_in = 32, channels_out = 64, kernel = 8, stride = 1, dropout = 0.5),
            conv_block(channels_in = 64, channels_out = 128, kernel = 8, stride = 1, dropout = 0.5),
            nn.Flatten()
            )

        out_shape = conv1D_out_shape(input_size, [8,2,8,2,8,2], [8,2,1,2,1,2], [4,0,4,0,4,0])

        self.TimeCrossSpace = nn.Sequential(
            nn.Linear(in_features=out_shape*128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),            
        )
        self.FreqCrossSpace = nn.Sequential(
            nn.Linear(in_features=out_shape*128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),            
        )

        if self.classify:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=256, out_features=64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=num_classes),
                nn.Softmax(dim = -1)
            )

    def forward(self, x_t, x_f):
        h_t = self.TimeEncoder(x_t)
        z_t = self.TimeCrossSpace(h_t)
        h_f = self.FrequencyEncoder(x_f)
        z_f = self.FreqCrossSpace(h_f)
        if self.classify:
            out = self.classifier(torch.cat([z_t, z_f], dim = -1))
            return h_t, z_t, h_f, z_f, out

        return h_t, z_t, h_f, z_f



class ContrastiveLoss(nn.Module):
    def __init__(self, tau, batchsize, device):
        super().__init__()
        self.tau = tau
        self.batchsize = batchsize
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim = -1)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')

        # compute negative mask
        upper = np.eye(2*batchsize, 2*batchsize, k = batchsize)
        lower = np.eye(2*batchsize, 2*batchsize, k = -batchsize)
        diag = np.eye(2*batchsize)

        self.negative_mask = torch.from_numpy(1-(upper+lower+diag)).bool().to(device)

    def forward(self, z_orig, z_augment):
        collect_z = torch.cat([z_augment, z_orig], dim = 0)
        # calculate cosine similarity between all augmented and
        # non-augmented latent representations
        similarities = self.cosine_similarity(collect_z.unsqueeze(1), collect_z.unsqueeze(0))

        # get the positive samples (upper and lower diagonal)
        upper_pos = torch.diag(similarities, self.batchsize)
        lower_pos = torch.diag(similarities, -self.batchsize)
        positive = torch.cat((upper_pos, lower_pos), dim = 0).unsqueeze(1)

        # get the negative samples by masking out the diagonal and the upper 
        # and lower diagonal
        negative = similarities[self.negative_mask].view(2*self.batchsize, -1)

        # concatenate the logits
        logits = torch.cat((positive, negative), dim = 1)
        logits /= self.tau
        
        # calculate loss using cross entropy by setting the "labels" to zero:
        # we have the positive samples on the 0th index in the logits and all 
        # other samples on the remaining indices - i.e. force the 0th "class"
        # to be large and all other to be smal
        labels = torch.zeros(2*self.batchsize).to(self.device).long()
         
        loss = self.criterion(logits, labels)
        return loss / (2*self.batchsize)
        

class TimeFrequencyLoss(nn.Module):
    def __init__(self, delta):
        self.delta = delta
    
