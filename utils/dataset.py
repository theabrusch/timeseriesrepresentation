from torch.utils.data import Dataset
import torch
from utils.augmentations import frequency_augmentation, time_augmentation

class TFC_Dataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X_t = X
        self.Y = Y.unsqueeze(1)
        self.channels = X.shape[1]
        self.time_length = X.shape[2]
        self.num_classes = len(torch.unique(Y))

        self.X_f = torch.fft.fft(self.X_t, axis = -1).abs()
        self.X_f_aug = frequency_augmentation(self.X_f, keep_all = False, abs_budget=True, return_ifft=False)
        self.X_t_aug = time_augmentation(self.X_t, keep_all = False, return_fft=False)
    
    def __getitem__(self, idx):
        freq_aug = torch.randint(high = 2, size = [1])
        time_aug = torch.randint(high = 3, size = [1])
        return self.X_t[idx], self.X_f[idx], self.X_t_aug[idx][time_aug].squeeze(1), self.X_f_aug[idx][freq_aug].squeeze(1), self.Y[idx]
    
    def __len__(self):
        return len(self.X_t)