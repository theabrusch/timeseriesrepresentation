import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import os
import glob
import mne 
from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import EpochTorchRecording, Thinker, Dataset, DatasetInfo
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def construct_eeg_datasets(config_path, dset, batchsize, sample_subjects = False, finetune = False, exclude_subjects = None):
    experiment = ExperimentConfig(config_path)
    config = experiment.datasets[dset]
    if finetune:
        config.chunk_duration = str(config.tlen)
    if not exclude_subjects is None:
        config.exclude_people = exclude_subjects

    thinkers = load_thinkers(config, sample_subjects=sample_subjects)
    info = DatasetInfo(config.name, config.data_max, config.data_min, config._excluded_people,
                        targets=config._targets if config._targets is not None else len(config._unique_events))

    train_thinkers, test_thinkers, val_thinkers = divide_thinkers(thinkers, finetune)
    train_dset, val_dset, test_dset = Dataset(train_thinkers, dataset_info=info), Dataset(val_thinkers, dataset_info=info), Dataset(test_thinkers, dataset_info=info)

    aug_config = { 
        'jitter_scale_ratio': 1.1,
        'jitter_ratio': 0.8,
        'max_seg': 8
    }
    train_dset, val_dset, test_dset = EEG_dataset(train_dset, aug_config), EEG_dataset(val_dset, aug_config), EEG_dataset(test_dset, aug_config)
    train_loader, val_loader, test_loader = DataLoader(train_dset, batch_size=batchsize), DataLoader(val_dset, batch_size=batchsize), DataLoader(test_dset, batch_size=batchsize)
    return train_loader, val_loader, test_loader, list(thinkers.keys()), (len(config.picks), config.tlen*train_dset.dn3_dset.sfreq, len(config.events.keys()))

def divide_thinkers(thinkers, finetune):
    if finetune:
        test_ratio = 0.5
    else:
        test_ratio = 0.3
    train, val_test = train_test_split(list(thinkers.keys()), test_size = test_ratio, random_state=0)
    val, test = train_test_split(val_test, test_size = 0.5, random_state=0)
    train_thinkers = dict()
    for subj in train:
        train_thinkers[subj] = thinkers[subj]
    
    val_thinkers = dict()
    for subj in val:
        val_thinkers[subj] = thinkers[subj]
    
    test_thinkers = dict()
    for subj in test:
        test_thinkers[subj] = thinkers[subj]
    return train_thinkers, test_thinkers, val_thinkers


def load_thinkers(config, sample_subjects = False):
    subjects = os.listdir(config.toplevel)
    subjects = [subject for subject in subjects if not subject in config.exclude_people]
    if sample_subjects:
        subjects = np.random.choice(subjects, sample_subjects)
    thinkers = dict()
    for i, subject in enumerate(subjects):
        print('Loading subject', i+1, 'of', len(subjects))
        subj_path = os.path.join(config.toplevel, subject)
        files = glob.glob(f'{subj_path}/*.fif')
        sessions = dict()
        for file in files:
            sess = file.split('/')[-1].strip('.fif')
            recording = construct_epoch_dset(file, config)
            sessions[sess] = recording
        if len(sessions.keys()) > 0:
            thinkers[subject] = Thinker(sessions)
    return thinkers


def construct_epoch_dset(file, config):
    raw = mne.io.read_raw_fif(file)
    sfreq = raw.info['sfreq']
    event_map = {v: v for v in config.events.values()}
    events = mne.events_from_annotations(raw, event_id=config.events, chunk_duration=eval(config.chunk_duration))[0]
    epochs = mne.Epochs(raw, events, tmin=config.tmin, tmax=config.tmin + config.tlen - 1 / sfreq, preload=config.preload, decim=config.decimate,
                        baseline=config.baseline, reject_by_annotation=config.drop_bad)
    recording = EpochTorchRecording(epochs, ch_ind_picks=config.picks, event_mapping=event_map,
                                    force_label=True)
    return recording


#def construct_dataset(config):

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma = 1.1):
    factor = np.random.normal(2, sigma, size = [1, x.shape[1]])
    return x * factor 

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments)

    ret = np.zeros_like(x)

    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        warp = np.concatenate(np.random.permutation(splits)).ravel()
        ret = x[:,warp]
    else:
        ret = x

    return ret


def remove_frequency(x, maskout_ratio=0):
    mask = torch.FloatTensor(x.shape).uniform_() > maskout_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0,):

    mask = torch.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix

class EEG_dataset(TorchDataset):
    def __init__(self, dn3_dset, augmentation_config, preloaded = False, fine_tune_mode = False):
        super().__init__()
        self.dn3_dset = dn3_dset
        self.aug_config = augmentation_config
        self.preloaded = preloaded
        self.fine_tune_mode = fine_tune_mode
    
    def _time_augmentations(self, signal):
        time_augs = [jitter, scaling, permutation]
        time_args = [self.aug_config['jitter_ratio'], self.aug_config['jitter_scale_ratio'], self.aug_config['max_seg']]
        li = np.random.randint(0, 3)
        return time_augs[li](signal, time_args[li])
    
    def _freq_augmentations(self, signal):
        freq_augs = [remove_frequency, add_frequency]
        li = np.random.randint(0, 2)
        return freq_augs[li](signal)

    def __len__(self):
        return len(self.dn3_dset)
    
    def _perform_augmentations(self, signal, fft):
        time_aug = self._time_augmentations(signal)
        freq_aug = self._freq_augmentations(fft)

        return time_aug, freq_aug
    
    def __getitem__(self, index):
        signal, label = self.dn3_dset.__getitem__(index)
        fft = torch.fft.fft(signal, axis = -1).abs()
        if not self.fine_tune_mode:
            time_aug, freq_aug = self._perform_augmentations(signal, fft)
            return signal, fft, time_aug, freq_aug, label
        else:
            return signal, fft, label
