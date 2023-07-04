import mne
import matplotlib.pyplot as plt

path = "/Users/theb/Desktop/training_raw/tr03-0005/001_30s_raw.fif"
raw = mne.io.read_raw_fif(path, preload=True)

raw = raw.filter(l_freq=0.5, h_freq=10)
raw = raw.resample(20)
data = raw.get_data(start = 100000, stop = 100000+1000)

for i in range(6):
    plt.figure()
    plt.plot(data[i], color = 'black', linewidth = 0.5)
    plt.axis('off')
    plt.savefig(f'/Users/theb/Documents/PhD/project/timeseriesrep/eeg_{i}.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0, transparent = True)
    plt.clf()



_LEFT_NUMBERS = list(reversed(range(1, 9, 2)))
_RIGHT_NUMBERS = list(range(2, 10, 2))

_EXTRA_CHANNELS = 5

DEEP_1010_CHS_LISTING = [
    # EEG
    "NZ",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F9", *["F{}".format(n) for n in _LEFT_NUMBERS], "FZ", *["F{}".format(n) for n in _RIGHT_NUMBERS], "F10",

    "FT9", "FT7", *["FC{}".format(n) for n in _LEFT_NUMBERS[1:]], "FCZ",
    *["FC{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "FT8", "FT10",
                                                                                                                                  
    "T9", "T7", "T3",  *["C{}".format(n) for n in _LEFT_NUMBERS[1:]], "CZ",
    *["C{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "T4", "T8", "T10",

    "TP9", "TP7", *["CP{}".format(n) for n in _LEFT_NUMBERS[1:]], "CPZ",
    *["CP{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "TP8", "TP10",

    "P9", "P7", "T5",  *["P{}".format(n) for n in _LEFT_NUMBERS[1:]], "PZ",
    *["P{}".format(n) for n in _RIGHT_NUMBERS[:-1]],  "T6", "P8", "P10",

    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1",  "OZ", "O2",
    "IZ",
    # EOG
    "VEOGL", "VEOGR", "HEOGL", "HEOGR",

    # Ear clip references
    "A1", "A2", "REF",
    # SCALING
    "SCALE",
    # Extra
    *["EX{}".format(n) for n in range(1, _EXTRA_CHANNELS+1)]
]