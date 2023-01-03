from utils.augmentations import time_augmentation, frequency_augmentation
import numpy as np
import torch

x = torch.randn(size = (100, 6, 1500))
x_aug = time_augmentation(x, keep_all = False, return_fft=False)