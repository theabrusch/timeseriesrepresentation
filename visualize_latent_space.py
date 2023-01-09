import pickle
import matplotlib.pyplot as plt
import numpy as np
import umap

path = 'outputs/classifier_False_TFC_True_abs_budget_True_stride_1_loss_poly/'
with open(f'{path}post_finetune_latent_variables.pickle', 'rb') as path:
    outputs = pickle.load(path) 

transform = umap.UMAP(n_neighbors=20, min_dist = 0.2, metric = 'cosine').fit_transform(np.concatenate((outputs['z_t'], outputs['z_f']), axis = 0))
transform_t = transform[:int(len(transform)/2),:]
transform_f = transform[int(len(transform)/2):,:]

#transform_f = umap.UMAP(n_neighbors=20, min_dist = 0.2, metric = 'cosine').fit_transform(outputs['z_f'])

colors = ['red', 'blue', 'green', 'yellow']
#colors = ['blue', 'blue', 'blue', 'blue']
color = [colors[i] for i in outputs['y']]

plt.scatter(transform_t[:,0], transform_t[:,1], marker = 'o', c = color)

plt.scatter(transform_f[:,0], transform_f[:,1], marker = 's', c = color)
plt.show()