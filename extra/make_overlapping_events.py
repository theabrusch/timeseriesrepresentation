import mne
from src.eegdataset import EpochTorchRecording
from dn3.configuratron import ExperimentConfig
import numpy as np


finetune_path = '/Users/theb/Documents/PhD/code/timeseriesrepresentation/sleepeeg_local.yml'
experiment = ExperimentConfig(finetune_path)
dset = 'sleepeeg'
config = experiment.datasets[dset]

file = '/Users/theb/Desktop/data/training_raw/tr03-0005/001_30s_raw.fif'
raw = mne.io.read_raw_fif(file, preload = True)

#if config.name == 'sleepedf':
#    annotations = raw.annotations
#    start_crop = annotations.orig_time + timedelta(seconds=annotations[1]['onset']) - timedelta(minutes=30)
#    end_crop = annotations.orig_time + timedelta(seconds=annotations[-2]['onset']) + timedelta(minutes=30)
#    annotations.crop(start_crop, end_crop)
#    raw.set_annotations(annotations, emit_warning=False)

def create_overlapping_events(raw, event_id, chunk_duration, overlap):
    annotations = raw.annotations
    descriptions = annotations.description

    event_sel = [ii for ii, kk in enumerate(descriptions) if kk in event_id]

    return events

sfreq = raw.info['sfreq']
event_map = {v: v for v in config.events.values()}


events = mne.events_from_annotations(raw, event_id=config.events, chunk_duration=eval(config.chunk_duration))[0]

epochs = mne.Epochs(raw, events, tmin=config.tmin, tmax=config.tmin + 32 - 1 / sfreq, preload=config.preload, decim=config.decimate,
                    baseline=config.baseline, reject_by_annotation=config.drop_bad)
recording = EpochTorchRecording(epochs, ch_ind_picks=config.picks, event_mapping=event_map,
                                force_label=True)