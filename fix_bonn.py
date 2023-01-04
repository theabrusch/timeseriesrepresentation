import numpy as np
import glob
import pickle

folders = glob.glob('datasets/Bonn/*')
file = np.loadtxt('datasets/Bonn/F/F002.txt')

train_data = np.zeros((500, 4097))
train_test = np.zeros(500)
labels = []
classes = []
i = 0 
for folder in folders:
    paths = glob.glob(folder + '/*.txt')
    class_ = folder.split('/')[-1]
    label = int(class_ == 'S')
    if class_ == 'N':
        paths = glob.glob(folder + '/*.TXT')
    for path in paths:
        file = np.loadtxt(path)
        train_data[i,:] = file
        train_test[i] = np.random.choice([0,1], p = [1/3, 2/3])
        labels.append(label)
        classes.append(class_)
        i+=1
labels = np.array(labels)
classes = np.array(classes)

collected_data = {
    'train': {
        'data': train_data[train_test == 1,:],
        'label': labels[train_test == 1],
        'class': classes[train_test == 1]
    },
    'test': {
        'data': train_data[train_test == 0,:],
        'label': labels[train_test == 0],
        'class': classes[train_test == 0]
    }
}

with open('datasets/Bonn/data.pkl', 'wb') as file:
    pickle.dump(collected_data, file)

train = train_data[train_test == 1,:]