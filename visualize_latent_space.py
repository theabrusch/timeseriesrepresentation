import pickle
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
temp = None
path = 'outputs/classifier_False_TFC_True_abs_budget_True_stride_1_loss_poly/'
finetune = 'post'
with open(f'{path}pretrain_latent_variables.pickle', 'rb') as file:
    outputs_pretrain = pickle.load(file) 
with open(f'{path}{finetune}_finetune_train_latent_variables.pickle', 'rb') as file:
    outputs_train = pickle.load(file) 

with open(f'{path}{finetune}_finetune_test_latent_variables.pickle', 'rb') as file:
    outputs_test = pickle.load(file) 

with open(f'{path}{finetune}_finetune_val_latent_variables.pickle', 'rb') as file:
    outputs_val = pickle.load(file) 

with open(f'{path}finetune_results.pickle', 'rb') as file:
    results = pickle.load(file) 

transform = umap.UMAP(n_neighbors=20, min_dist = 0.2, metric = 'cosine').fit_transform(np.concatenate((outputs_pretrain['z_t'][idx], outputs_pretrain['z_f'][idx]), axis = 0))
transform_t = transform[:int(len(transform)/2),:]
transform_f = transform[int(len(transform)/2):,:]

colors = ['red', 'blue', 'green', 'yellow']
#colors = ['blue', 'blue', 'blue', 'blue']
color = [colors[i] for i in outputs_pretrain['y']]
color_test = [colors[i] for i in outputs_test['y']]
color_val = [colors[i] for i in outputs_val['y']]

idx = np.arange(len(outputs_pretrain['z_t']))
idx = np.random.choice(idx, size = 1000)

plt.scatter(transform_t[idx,0], transform_t[idx,1], marker = 'o', c = np.array(color)[idx], label = 'time embeddings')
plt.scatter(transform_f[idx,0], transform_f[idx,1], marker = 's', c = np.array(color)[idx], label = 'frequency embeddings')
plt.legend()
plt.show()

plt.plot(outputs_test['x'][6,0,:])
plt.show()

val_accuracy = []
test_accuracy = []
input_type = 'time'
if input_type == 'both':
    train_input = np.concatenate((outputs_train['z_t'], outputs_train['z_f']), axis = 1)
    val_input = np.concatenate((outputs_val['z_t'], outputs_val['z_f']), axis = 1)
    test_input = np.concatenate((outputs_test['z_t'], outputs_test['z_f']), axis = 1)
elif input_type == 'freq':
    train_input = outputs_train['z_f']
    val_input = outputs_val['z_f']
    test_input = outputs_test['z_f']
else:
    train_input = outputs_train['z_t']
    val_input = outputs_val['z_t']
    test_input = outputs_test['z_t']

classifier = 'knn'

if classifier == 'knn':
    # train KNeighborsClassifier
    for neighbors in range(1,10):
        classifier = KNeighborsClassifier(n_neighbors=neighbors)
        classifier.fit(train_input, outputs_train['y'])
        val_out = classifier.predict(val_input)
        val_accuracy.append(accuracy_score(outputs_val['y'], val_out))
        test_out = classifier.predict(test_input)
        test_accuracy.append(accuracy_score(outputs_test['y'], test_out))
else:
    classifier = LogisticRegression()
    classifier.fit(train_input, outputs_train['y'])
    val_out = classifier.predict(val_input)
    val_accuracy = accuracy_score(outputs_val['y'], val_out)
    test_out = classifier.predict(test_input)
    test_accuracy = accuracy_score(outputs_test['y'], test_out)
print('Validation accuracy', val_accuracy)
print('Test accuracy', test_accuracy)


umapper = umap.UMAP(n_neighbors=20, min_dist = 0.2, metric = 'cosine')
transform_train = umapper.fit_transform(np.concatenate((outputs_train['z_t'], outputs_train['z_f']), axis = 1))
transform_test = umapper.transform(np.concatenate((outputs_test['z_t'], outputs_test['z_f']), axis = 1))
transform_val = umapper.transform(np.concatenate((outputs_val['z_t'], outputs_val['z_f']), axis = 1))

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(transform_train, outputs_train['y'])
val_out = classifier.predict(transform_val)
val_accuracy = accuracy_score(outputs_val['y'], val_out)
test_out = classifier.predict(transform_test)
test_accuracy = accuracy_score(outputs_test['y'], val_out)

plt.scatter(transform_train[:,0], transform_train[:,1], marker = 'o', c = color, label = 'train embeddings')
plt.scatter(transform_test[:,0], transform_test[:,1], marker = 's', c = color_test, label = 'test embeddings')
plt.scatter(transform_val[:,0], transform_val[:,1], marker = 'v', c = color_val, label = 'val embeddings')
plt.legend()
plt.show()

