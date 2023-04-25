import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rc
# Set font family to Times New Roman

api = wandb.Api()

optenc = False
methods = ['GNN', 'COCOA']
learning_rates = {'GNN': [5e-4], 'COCOA': [3e-3]}
losses = ['COCOA', 'contrastive', 'time_loss', 'scratch']

all_results = dict()
for method in methods:
    all_results[method] = dict()
    for lr in learning_rates[method]:
        all_results[method][lr] = dict()
        for loss in losses:
            all_results[method][lr][loss] = dict()
            all_results[method][lr][loss]['val_acc'] = []
            all_results[method][lr][loss]['test_acc'] = []
            all_results[method][lr][loss]['n_samples'] = []
            runs = api.runs(path = 'theabrusch/MultiView', filters ={"$and": [{"group": f'{method}_{loss}'}, {'config.ft_learning_rate': lr}, {'config.optimize_encoder': optenc}]})
            print(f'{method}_{loss}')
            print(f'runs: {len(runs)}\n')
            for run in runs:
                hist = run.history()
                best_val_loss_arg = np.argmin(hist['val_class_loss'])
                best_val_acc = hist['val_acc'][best_val_loss_arg]
                best_test_acc = hist['test_acc'][best_val_loss_arg]
                n_samples = run.config['Finetune validation samples']

                all_results[method][lr][loss]['val_acc'].append(best_val_acc)
                all_results[method][lr][loss]['test_acc'].append(best_test_acc)
                all_results[method][lr][loss]['n_samples'].append(n_samples)

# print average validation accuracy for each method, learning rate and loss
for method in methods:
        for loss in losses:
            for lr in learning_rates[method]:
                print(f'{method}_{loss}')
                print(f'lr: {lr}')
                print(f'val acc: {np.mean(all_results[method][lr][loss]["val_acc"])}')
                print(f'test acc: {np.mean(all_results[method][lr][loss]["test_acc"])}')
                print(f'n_samples: {np.mean(all_results[method][lr][loss]["n_samples"])}\n')

# plot results in same figure
fig, ax = plt.subplots(1, 1, figsize = (8, 4))
ax.set_xlabel('Number of samples per class', fontsize = 14)
ax.set_ylabel('Test accuracy (%)', fontsize = 14)
ax.set_title('Test accuracy vs. number of samples', fontsize = 14)
ax.set_ylim(15, 70)
# make x axis logarithmic
ax.set_xscale('log')
rc('font',**{'family':'serif','serif':['Times']})

method_names = {'GNN': ', GNN', 'COCOA': ''}
methods = ['COCOA', 'GNN']

learning_rates = {'GNN': 5e-4, 'COCOA': 3e-3}
loss_names = {'scratch': 'Scratch', 'contrastive': 'NT-Xent', 'time_loss': 'TS2Vec', 'COCOA': 'COCOA'}
# define 8 different markers for each method and loss
markers = ['o', 'v', 's', 'p', '*', 'h', 'D', 'X']
# format test accuarcy as latex results
i = 0
for loss in losses:
    for method in methods:
        print(method, loss)
        # sort test accuracy by n_samples
        idx = np.argsort(all_results[method][learning_rates[method]][loss]['n_samples'])
        test_accuracy = np.array(all_results[method][learning_rates[method]][loss]['test_acc'])[idx]
        n_samples = np.array(all_results[method][learning_rates[method]][loss]['n_samples'])[idx]
        # plot with marker on top of line
        ax.plot(n_samples/5, test_accuracy*100, label = f'{loss_names[loss]}{method_names[method]}', marker = markers[i], markersize = 6)
        sub_idx = [0,1,2,-1]
        print('n_samples', n_samples[sub_idx])
        test_str = ' & '.join([f'${acc*100:.1f}\%$' for acc in test_accuracy[sub_idx]])
        print('test accuracy', test_str, '\n')
        i+=1

ax.legend(loc = 'upper center', ncol = 4, fontsize = 12, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

print(plt.style.available)