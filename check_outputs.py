import matplotlib.pyplot as plt
import pickle
import datetime

with open('outputs.pickle', 'rb') as path:
    output = pickle.load(path)

plt.plot(output['train']['time_loss'], label = 'Time loss')
plt.plot(output['train']['freq_loss'], label = 'Freq. loss')
plt.plot(output['train']['time_freq_loss'], label = 'Time-Freq. loss')
plt.plot(output['train']['loss'], label = 'Total loss')
plt.legend()
plt.show()

print(datetime.datetime.now())