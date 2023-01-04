from utils.augmentations import time_augmentation, frequency_augmentation
from utils.dataset import TFC_Dataset
import numpy as np
import torch

data_path = 'datasets/Epilepsy/'
data = torch.load(data_path + 'train.pt')
x = data['samples']
y = data['labels']

x_aug = time_augmentation(x, keep_all = False, return_fft=False)
x_f = torch.fft.fft(x, dim = -1)
x_f_aug = frequency_augmentation(x_f, keep_all=False, return_ifft=False)

dset = TFC_Dataset(x, y)

temp = dset.__getitem__(1)