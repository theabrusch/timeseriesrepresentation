from torch.utils.data import Dataset
import torch
from utils.augmentations import frequency_augmentation, time_augmentation
from torch.nn import functional as F

class TFC_Dataset(Dataset):
    def __init__(self, X, Y, dset, sample_channel = False, abs_budget = False, fine_tune_mode = False, test_mode = False):
        super().__init__()

        if dset == 'HAR' and not sample_channel:
            X = F.pad(X, (0,int(206-X.shape[2])))
            if not sample_channel:
                self.channels = 3
                X = X[:,:3,:]
            else:
                self.channels = X.shape[1]
        else:
            self.channels = X.shape[1]

        self.X_t = X
        self.Y = Y
        self.time_length = X.shape[2]
        self.num_classes = len(torch.unique(Y))
        self.sample_channel = sample_channel
        if int(torch.max(self.Y)) == self.num_classes:
            self.Y = self.Y-1
        self.test_mode = test_mode
        self.fine_tune_mode = fine_tune_mode

        self.X_f = torch.fft.fft(self.X_t, axis = -1).abs()
        if not test_mode and not fine_tune_mode:
            self.X_f_aug = frequency_augmentation(self.X_f, keep_all = False, abs_budget=abs_budget, return_ifft=False)
            self.X_t_aug = time_augmentation(self.X_t, keep_all = False, return_fft=False)
    
    def __getitem__(self, idx):
        freq_aug = torch.randint(high = 2, size = [1])
        time_aug = torch.randint(high = 3, size = [1])
        if not self.test_mode and not self.fine_tune_mode:
            if not self.sample_channel:
                return self.X_t[idx], self.X_f[idx], self.X_t_aug[idx][time_aug], self.X_f_aug[idx][freq_aug], self.Y[idx]
            else:
                ch = torch.randint(high = self.channels, size = [1])
                return self.X_t[idx][ch,:].unsqueeze(0), self.X_f[idx][ch,:].unsqueeze(0), self.X_t_aug[idx][time_aug,ch,:].unsqueeze(0), self.X_f_aug[idx][freq_aug,ch,:].unsqueeze(0), self.Y[idx]
        elif self.fine_tune_mode:
            return self.X_t[idx], self.X_f[idx], self.X_t[idx], self.X_f[idx], self.Y[idx]
        else:
            return self.X_t[idx], self.X_f[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X_t)