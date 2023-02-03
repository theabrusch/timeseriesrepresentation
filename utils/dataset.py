from torch.utils.data import Dataset, DataLoader
import torch
from utils.augmentations import frequency_augmentation, time_augmentation
from torch.nn import functional as F

def get_datasets(data_path, abs_budget, batch_size, finetune_mode = False, sample_channel = False):
    train = torch.load(data_path + 'train.pt')
    val = torch.load(data_path + 'val.pt')
    test = torch.load(data_path + 'test.pt')

    dset = data_path.split('/')[-2]
    TFC_dset = TFC_Dataset(train['samples'], train['labels'], dset, abs_budget=abs_budget, fine_tune_mode=finetune_mode, sample_channel=sample_channel)
    train_loader = DataLoader(TFC_dset, batch_size = batch_size, shuffle = True, drop_last=False)

    val_dset = TFC_Dataset(val['samples'], val['labels'], dset = dset, abs_budget = abs_budget, fine_tune_mode=finetune_mode, sample_channel=sample_channel)
    test_dset = TFC_Dataset(test['samples'], test['labels'], dset = dset, test_mode = True, fine_tune_mode=False, sample_channel=sample_channel)
    val_loader = DataLoader(val_dset, batch_size = batch_size, drop_last=False)
    test_loader = DataLoader(test_dset, batch_size = batch_size, drop_last=False)

    return train_loader, val_loader, test_loader, (TFC_dset.channels, TFC_dset.time_length, TFC_dset.num_classes)

def get_dset_info(data_path = None, X = None, dset = None, sample_channel=False):
    if X is None:
        train = torch.load(data_path + 'train.pt')
        X = train['samples']
        dset = data_path.split('/')[-2]
    
    if dset == 'HAR':
        time_length = 206
        if not sample_channel:
            channels = 3
        else:
            channels = X.shape[1]
    else:
        time_length = X.shape[2]
        channels = X.shape[1]
    return channels, time_length   
    


class TFC_Dataset(Dataset):
    def __init__(self, X, Y, dset, sample_channel = False, abs_budget = False, fine_tune_mode = False, test_mode = False):
        super().__init__()

        channels, time_length = get_dset_info(X = X, dset = dset, sample_channel = sample_channel)

        if time_length > X.shape[2]:
             X = F.pad(X[:,:channels,:], (0,int(206-X.shape[2])))
        else:
            X = X[:,:channels,:time_length]

        self.X_t = X
        self.Y = Y
        self.time_length = X.shape[2]
        self.num_classes = len(torch.unique(Y))
        
        if not sample_channel:
            self.channels = channels
        else:
            self.num_channels = channels
            # num of channels for the NN to take as input
            self.channels = 1
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
            return self.X_t[idx], self.X_f[idx], self.X_t_aug[idx][time_aug].squeeze(0), self.X_f_aug[idx][freq_aug].squeeze(0), self.Y[idx]
        elif self.fine_tune_mode:
            return self.X_t[idx], self.X_f[idx], self.X_t[idx], self.X_f[idx], self.Y[idx]
        else:
            return self.X_t[idx], self.X_f[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X_t)