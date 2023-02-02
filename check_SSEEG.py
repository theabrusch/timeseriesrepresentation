from scipy.io import loadmat, savemat
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.signal
import os

def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

def extract_labels(path):
    data = h5py.File(path, 'r')
    length = data['data']['sleep_stages']['wake'].shape[1]
    labels = np.zeros((length, 6)) 

    for i, label in enumerate(data['data']['sleep_stages'].keys()):
        labels[:,i] = data['data']['sleep_stages'][label][:]
    
    return labels

def resample_signal(data, labels, old_fs):
    diff = np.diff(labels, axis = 0)
    cutoff = np.where(diff[:,4] != 0)[0]+1
    data, labels = data[cutoff[0]+1:,:], labels[cutoff[0]+1:,:]

    new_fs = 100
    num = int(len(data)/(old_fs/new_fs))
    resampled_data = scipy.signal.resample(data, num = num, axis = 0)
    resampled_labels = labels[::int((old_fs/new_fs)),:]
    return resampled_data, resampled_labels


def preprocess_EEG(folder,remove_files = False, out_folder = None):
    files = glob.glob(f'{folder}/*')
    data = None
    labels = None
    Fs = None
    for file in files:
        if '.hea' in file:
            s, Fs, n_samples = import_signal_names(file)
            if remove_files:
                os.remove(file)
        elif '-arousal.mat' in file:
            labels = extract_labels(file)
            if remove_files:
                os.remove(file)
        elif 'mat' in file:
            data = loadmat(file)['val'][:6, :].T
            if remove_files:
                os.remove(file)

    if not data is None:
        resampled_data, resampled_labels = resample_signal(data, labels, old_fs = Fs)
        if not out_folder is None:
            hdf5_path = f'{out_folder}/data.hdf5'
        else:
            hdf5_path = f'{folder}/data.hdf5'

        hdf5_file = h5py.File(hdf5_path, 'w')
        hdf5_data = hdf5_file.create_group("data")
        _ = hdf5_data.create_dataset('EEG', data=resampled_data.astype(np.int16))
        _ = hdf5_data.create_dataset('labels', data=resampled_labels.astype(int))

        hdf5_file.close()

def relocate_EEG_data(folder, remove_files = True):
    data_file = h5py.File(f'{folder}/data.hdf5', 'r')
    data = data_file["data"]["EEG"][:]
    labels = data_file["data"]["labels"][:]
    collect_data = {"data": data, "labels": labels}
    savemat(f'{folder}/data.mat', collect_data)
    




out_folder = '/Users/theb/Desktop/physionet.org/physionet.org/files/challenge-2018/1.0.0/training/'
root_folder = '/Volumes/SED/training/'
subjects = os.listdir(root_folder)

for i, subject in enumerate(subjects):
    print('Processing subject', i+1, 'of', len(subjects))
    subj_folder = os.path.join(root_folder, subject)
    try:
        preprocess_EEG(subj_folder, out_folder = out_folder, remove_files = False)
    except:
        print('Issue with subject', subject)