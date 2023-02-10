import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import os
import glob
import json
import mne 
from dn3.configuratron import ExperimentConfig
from dn3.data.dataset import EpochTorchRecording, Thinker, Dataset, DatasetInfo
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def construct_eeg_datasets(config_path, 
                           finetune_path,
                           batchsize, 
                           target_batchsize,
                           normalize = False,
                           sample_subjects = False, 
                           sample_test_subjects = False,
                           exclude_subjects = None,
                           train_mode = 'both'):
    experiment = ExperimentConfig(config_path)
    dset = config_path.split('/')[-1].strip('.yml').split('_')[0]
    config = experiment.datasets[dset]
    config.normalize = normalize

    if not exclude_subjects is None:
        config.exclude_people = exclude_subjects
    
    if finetune_path == 'same':
        split_path = config_path.removesuffix('.yml') + '_splits.txt'
        with open(split_path, 'r') as split_file:
            splits = json.load(split_file)
        pretrain_subjects = splits['pretrain']
    else:
        subjects = None

    # construct pretraining datasets
    if train_mode == 'pretrain' or train_mode == 'both':
        print('Loading pre-training data')
        pretrain_thinkers = load_thinkers(config, sample_subjects=sample_subjects, subjects = pretrain_subjects)
        info = DatasetInfo(config.name, config.data_max, config.data_min, config._excluded_people,
                            targets=config._targets if config._targets is not None else len(config._unique_events))
        pretrain_train_thinkers, pretrain_val_thinkers = divide_thinkers(pretrain_thinkers)
        pretrain_dset, pretrain_val_dset = Dataset(pretrain_train_thinkers, dataset_info=info), Dataset(pretrain_val_thinkers, dataset_info=info)

        aug_config = { 
            'jitter_scale_ratio': 1.1,
            'jitter_ratio': 0.8,
            'max_seg': 8
        }
        pretrain_dset, pretrain_val_dset = EEG_dataset(pretrain_dset, aug_config), EEG_dataset(pretrain_val_dset, aug_config)
        pretrain_loader, pretrain_val_loader = DataLoader(pretrain_dset, batch_size=batchsize), DataLoader(pretrain_val_dset, batch_size=batchsize)
    else:
        pretrain_loader, pretrain_val_loader = None, None
    
    # construct finetuning dataset
    if train_mode == 'finetune' or train_mode == 'both':
        sample_subjects = int(sample_subjects/2) if sample_subjects else sample_subjects
        if not finetune_path == 'same':
            experiment = ExperimentConfig(finetune_path)
            dset = config_path.split('/')[-1].strip('.yml')
            config = experiment.datasets[dset]
            finetunesubjects = None
            test_subjects = None
        else:
            config.chunk_duration = str(config.tlen)
            finetunesubjects = splits['finetune']
            test_subjects = splits['test']
        print('Loading finetuning data')
        finetune_thinkers = load_thinkers(config, sample_subjects=sample_subjects, subjects = finetunesubjects)
        finetune_train_thinkers, finetune_val_thinkers = divide_thinkers(finetune_thinkers)
        finetune_train_dset, finetune_val_dset = Dataset(finetune_train_thinkers, dataset_info=info), Dataset(finetune_val_thinkers, dataset_info=info)

        aug_config = { 
            'jitter_scale_ratio': 1.1,
            'jitter_ratio': 0.8,
            'max_seg': 8
        }
        finetune_train_dset, finetune_val_dset = EEG_dataset(finetune_train_dset, aug_config), EEG_dataset(finetune_val_dset, aug_config)
        finetune_loader, finetune_val_loader = DataLoader(finetune_train_dset, batch_size=target_batchsize), DataLoader(finetune_val_dset, batch_size=target_batchsize)
        # get test set
        print('Loading test data')
        test_thinkers = load_thinkers(config, sample_subjects = sample_test_subjects, subjects = test_subjects)
        test_dset = Dataset(test_thinkers, dataset_info=info)
        test_dset = EEG_dataset(test_dset, aug_config, fine_tune_mode=True)
        test_loader = DataLoader(test_dset, batch_size=batchsize)
    else:
        finetune_loader, finetune_val_loader, test_loader = None

    return pretrain_loader, pretrain_val_loader,finetune_loader, finetune_val_loader, test_loader, (len(config.picks), config.tlen*100, len(config.events.keys()))

def divide_thinkers(thinkers):
    train, val = train_test_split(list(thinkers.keys()), test_size = 0.2, random_state=0)
    train_thinkers = dict()
    for subj in train:
        train_thinkers[subj] = thinkers[subj]
    val_thinkers = dict()
    for subj in val:
        val_thinkers[subj] = thinkers[subj]

    return train_thinkers, val_thinkers


def load_thinkers(config, sample_subjects = False, subjects = None):
    if subjects is None:
        subjects = os.listdir(config.toplevel)        
    subjects = [subject for subject in subjects if not subject in config.exclude_people]
    if sample_subjects:
        np.random.seed(0)
        subjects = np.random.choice(subjects, sample_subjects, replace = False)
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
    raw = mne.io.read_raw_fif(file, preload = config.preload)
    if config.normalize and config.preload:
        raw.apply_function(lambda x: (x-np.mean(x))/np.std(x))
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
