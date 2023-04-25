import mne
import matplotlib.pyplot as plt

path = "/Users/theb/Desktop/training_raw/tr03-0005/001_30s_raw.fif"
raw = mne.io.read_raw_fif(path, preload=True)

raw = raw.filter(l_freq=0.5, h_freq=10)
raw = raw.resample(20)
data = raw.get_data(start = 200000, stop = 200000+1000)

for i in [1, 3, 5]:
    plt.figure()
    plt.plot(data[i], color = 'black', linewidth = 0.5)
    plt.axis('off')
    plt.savefig(f'/Users/theb/Documents/PhD/project/timeseriesrep/eeg_{i}.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0, transparent = True)
    plt.clf()