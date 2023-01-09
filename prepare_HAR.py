import numpy as np
import glob
from sklearn.model_selection import train_test_split
import torch

path = 'datasets/HAR/UCI HAR Dataset'
train_test = ['train', 'test']

j = 0
for set in train_test:
    set_path = f'{path}/{set}'

    y_temp = np.loadtxt(f'{set_path}/y_{set}.txt')

    signal_paths = np.sort(glob.glob(f'{set_path}/Inertial Signals/*.txt'))
    x_train = np.zeros((len(y_temp), 9, 128))
    for i, sig_path in enumerate(signal_paths):
        x_train[:,i,:] = np.loadtxt(sig_path)
    
    if j == 0:
        y_collect = y_temp
        x_collect = x_train
    else:
        y_collect = np.concatenate((y_collect, y_temp), axis = 0)
        x_collect = np.concatenate((x_collect, x_train), axis = 0)
    j+=1

X_train, X_val_test, y_train, y_val_test = train_test_split(x_collect, y_collect, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_val_test, y_val_test, test_size=0.5)

X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

train = {'samples': X_train, 'labels': y_train}
val = {'samples': X_val, 'labels': y_val}
test = {'samples': X_test, 'labels': y_test}

save_path = 'datasets/HAR'
torch.save(train, f'{save_path}/train.pt')
torch.save(val, f'{save_path}/val.pt')
torch.save(test, f'{save_path}/test.pt')