import mne
import os
import glob
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def preprocess_EEG(file, out_folder = None):

    raw = mne.io.read_raw_edf(file, stim_channel='Event marker', preload = True)
    file_split = file.split('/')
    subject = file_split[-1][:5]
    session = file_split[-1][5:7]
    file_path = '/'.join(file_split[:-1])
    
    anno_path = glob.glob(f'{file_path}/{subject}{session}*-Hypnogram.edf')[0]
    
    annot_train = mne.read_annotations(anno_path)
    raw.set_annotations(annot_train, emit_warning=False)
    out_path = f'{out_folder}/{subject}/'
    os.makedirs(out_path, exist_ok = True)
    raw.save(f'{out_path}{subject}{session}_raw.fif', overwrite = True)



#alice_files = ['/Users/theb/Desktop/sleep_edf/physionet-sleep-data/SC4001E0-PSG.edf', '/Users/theb/Desktop/sleep_edf/physionet-sleep-data/SC4001EC-Hypnogram.edf']
#raw_train = mne.io.read_raw_edf(alice_files[0], stim_channel='Event marker')
#annot_train = mne.read_annotations(alice_files[1])
#raw_train.set_annotations(annot_train, emit_warning=False)
#
#events = mne.events_from_annotations(raw_train, chunk_duration = 30)
#epochs = mne.Epochs(raw_train, events, tmin=0, tmax=30 - 1 / 100, baseline = None)

root_folder = '/Users/theb/Desktop/sleep_edf/physionet-sleep-data/*-PSG.edf'

subjects = glob.glob(root_folder)
ss = [subj.split('/')[-1][:5] for subj in subjects]
file = subjects[0]

file_split = file.split('/')
subject = file_split[-1][:5]
session = file_split[-1][5:7]
file_path = '/'.join(file_split[:-1])

anno_path = f'{file_path}/{subject}{session}C-Hypnogram.edf'
for i, subject in enumerate(subjects):
    print('Subject', i+1, 'of', len(subjects))
    preprocess_EEG(subject, out_folder = '/Users/theb/Desktop/sleep_edf/mne_files')
