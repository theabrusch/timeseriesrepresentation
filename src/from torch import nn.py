from torch.nn import functional as F
import torch

temp = torch.randn(2,1, 4)
temp2 = torch.randn(1, 2, 4)

sim = F.cosine_similarity(temp, temp2, dim=-1)