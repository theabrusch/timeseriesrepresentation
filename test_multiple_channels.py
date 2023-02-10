
from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import EpochTorchRecording, Thinker, DatasetInfo, Dataset
import mne
from dn3_.dn3.utils import make_epochs_from_raw
from eegdataset import load_thinkers, EEG_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.fft import fft
import numpy as np
import json
import os

config_filename = 'sleepeeg_local.yml'
experiment = ExperimentConfig(config_filename)
ds_config = experiment.datasets['sleepeeg']

subjects = os.listdir(ds_config.toplevel)
subjects = [subject for subject in subjects if 'tr' in subject]
train, test_val = train_test_split(subjects, train_size = 596)
test, val = train_test_split(test_val, train_size = 200)

splits = {'pretrain': train, 'finetune': val, 'test': test}

with open('sleepeeg_local_splits.txt', 'r') as split_file:
    splits = json.load(split_file)

thinkers = load_thinkers(ds_config, sample_subjects=20)

train, val_test = train_test_split(list(thinkers.keys()), test_size = 0.4)
val, test = train_test_split(val_test, test_size = 0.5)

info = DatasetInfo(ds_config.name, ds_config.data_max, ds_config.data_min, ds_config._excluded_people,
                           targets=ds_config._targets if ds_config._targets is not None else len(ds_config._unique_events))

dset = Dataset(thinkers = thinkers, dataset_info=info)

config = { 
        'jitter_scale_ratio': 1.1,
        'jitter_ratio': 0.8,
        'max_seg': 8
}

eeg_dset = EEG_dataset(dset, config)

targets = eeg_dset.dn3_dset.get_targets()

np.unique(targets, return_counts=True)
splits = dset.lmso(test_splits=[test], validation_splits=[val])

dloader = DataLoader(eeg_dset, batch_size=64)
temp = next(iter(dloader))

def mne_fft(signal):
    return abs(fft(signal, axis = -1))

def fft_thinker(id, thinker):
    for session in thinker.sessions.values():
        session.epochs.apply_function(mne_fft, pick = 'eeg', channel_wise = False)

dset._apply(fft_thinker)


path = '/Users/theb/Desktop/training_raw/tr03-0005/001_30s_raw.fif'
raw = mne.io.read_raw_fif(path)
#make_epochs_from_raw(raw: mne.io.Raw, tmin, tlen, event_ids=None, baseline=None, decim=1, filter_bp=None,
#                         drop_bad=False, use_annotations=False, chunk_duration=None)
epochs = make_epochs_from_raw(raw, tmin = 0, tlen = 30, chunk_duration = 30, use_annotations = True, event_ids = {'nonrem1': 1, 'nonrem2': 2, 'nonrem3': 3, 'rem': 4, 'wake': 5})

ds_config.deep1010 = None
ds_config.chunk_duration = None
dataset = ds_config.auto_construct_dataset()

dataset.apply()

dataset2 = ds_config.auto_construct_dataset()
data = dataset.to_numpy()