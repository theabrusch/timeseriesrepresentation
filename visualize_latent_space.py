import pickle
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
temp = None
path = 'outputs/TFC_pretrain_all_encoder_TFC_standardize_channelwise_multchannel_None_sleepeeg_v_9/'
finetune = 'prior'
finetune_dset = 'sleepeeg'
with open(f'{path}pretrain_latent_variables.pickle', 'rb') as file:
    outputs_pretrain = pickle.load(file) 

with open(f'{path}{finetune}_finetune_train_latent_variables_{finetune_dset}.pickle', 'rb') as file:
    outputs_train = pickle.load(file) 

with open(f'{path}{finetune}_finetune_test_latent_variables_{finetune_dset}.pickle', 'rb') as file:
    outputs_test = pickle.load(file) 

with open(f'{path}{finetune}_finetune_val_latent_variables_{finetune_dset}.pickle', 'rb') as file:
    outputs_val = pickle.load(file) 

with open(f'{path}finetune_results_optenc_True_{finetune_dset}.pickle', 'rb') as file:
    results = pickle.load(file) 

UM = umap.UMAP(n_neighbors=20, min_dist = 0.2, metric = 'cosine')
transform = UM.fit_transform(outputs_train['z_f'])
transform_t = transform
#transform_f = transform[int(len(transform)/2):,:]

transform_test = UM.transform(outputs_test['z_f'])
transform_t_test = transform_test
#transform_f_test = transform_test[int(len(transform_test)/2):,:]

transform_val = UM.transform(outputs_val['z_f'])
transform_t_val = transform_val

#transform_f_val = transform_val[int(len(transform_val)/2):,:]

colors = ['red', 'blue', 'green', 'yellow', 'purple', 'lightgreen', 'pink', 'black']
#colors = ['blue', 'blue', 'blue', 'blue']
color = [colors[int(i)] for i in outputs_train['y']]
color_test = [colors[int(i)] for i in outputs_test['y']]
color_val = [colors[int(i)] for i in outputs_val['y']]

idx = np.arange(len(outputs_train['z_t']))
#idx = np.random.choice(idx, size = 200)

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))
ax[0].scatter(transform_t[idx,0], transform_t[idx,1], marker = 'v', c = np.array(color)[idx], label = 'time embeddings')
#ax[0].scatter(transform_f[idx,0], transform_f[idx,1], marker = 's', c = np.array(color)[idx], label = 'frequency embeddings')
ax[0].set_title('Train set embeddings', fontsize = 14)

ax[0].legend(fontsize = 14)
ax[2].scatter(transform_t_test[:,0], transform_t_test[:,1], marker = 'v', c = np.array(color_test), label = 'time embeddings')
#ax[2].scatter(transform_f_test[:,0], transform_f_test[:,1], marker = 's', c = np.array(color_test), label = 'frequency embeddings')
ax[2].set_title('Test set embeddings', fontsize = 14)
ax[2].legend(fontsize = 14)

ax[1].scatter(transform_t_val[:,0], transform_t_val[:,1], marker = 'v', c = np.array(color_val), label = 'time embeddings')
#ax[1].scatter(transform_f_val[:,0], transform_f_val[:,1], marker = 's', c = np.array(color_val), label = 'frequency embeddings')
ax[1].set_title('Val. set embeddings', fontsize = 14)
ax[1].legend(fontsize = 14)
plt.tight_layout()
plt.show()

plt.plot(outputs_test['x'][6,0,:])
plt.show()

val_res = {
    'acc': [],
    'prec': [],
    'rec': [],
    'f1': [],
}

test_res = {
    'acc': [],
    'prec': [],
    'rec': [],
    'f1': [],
}

input_type = 'freq'
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
        val_res['acc'].append(accuracy_score(outputs_val['y'], val_out))
        prec, rec, f1, _ = precision_recall_fscore_support(outputs_val['y'], val_out)
        val_res['prec'].append(np.mean(prec))
        val_res['rec'].append(np.mean(rec))
        val_res['f1'].append(np.mean(f1))
        test_out = classifier.predict(test_input)
        test_res['acc'].append(accuracy_score(outputs_test['y'], test_out))
        prec, rec, f1, _ = precision_recall_fscore_support(outputs_test['y'], test_out)
        test_res['prec'].append(np.mean(prec))
        test_res['rec'].append(np.mean(rec))
        test_res['f1'].append(np.mean(f1))
else:
    classifier = LogisticRegression(max_iter = 10000)
    classifier.fit(train_input, outputs_train['y'])
    val_out = classifier.predict(val_input)
    val_accuracy = accuracy_score(outputs_val['y'], val_out)
    test_out = classifier.predict(test_input)
    test_accuracy = accuracy_score(outputs_test['y'], test_out)
print('Validation accuracy', val_res)
print('Test accuracy', test_res)


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
plt.legend(fontsize = 14)
plt.show()

