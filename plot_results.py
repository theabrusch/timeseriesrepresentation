import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rc
# Set font family to Times New Roman

api = wandb.Api()

optenc = True
methods = ['GNN', 'COCOA']
learning_rates = {'GNN': [5e-4], 'COCOA': [5e-4]}
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
                if 'seed' in run.config.keys():
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

method = 'GNN'
lr = 5e-4
loss = 'time_loss'
method_names = {'GNN': ', MPNN', 'COCOA': ''}
methods = ['COCOA', 'GNN']
losses = ['COCOA', 'contrastive', 'time_loss', 'scratch']
loss_names = {'scratch': 'Scratch', 'contrastive': 'NT-Xent', 'time_loss': 'TS2Vec', 'COCOA': 'COCOA'}
learning_rates = {'GNN': 5e-4, 'COCOA': 5e-4}

table = ''
idx = np.argsort(all_results[method][learning_rates[method]][loss]['n_samples'])
test_accuracy = np.array(all_results[method][learning_rates[method]][loss]['test_acc'])[idx]
print(test_accuracy.reshape(6,5))
test_accuracy = np.reshape(test_accuracy, (6,5)).mean(1)
n_samples = np.array(all_results[method][learning_rates[method]][loss]['n_samples'])[idx].reshape(6,5).mean(1)
print(n_samples)
sub_idx = [0,1,2,-1]
sub_acc = test_accuracy[sub_idx]
acc_str = [f'{s_acc:.3f}'[1:] for s_acc in sub_acc]
method_name = method_names[method]
table_line = f'{method_name} & {loss_names[loss]} & ${sub_acc[0]:.1f}\%$ & ${sub_acc[1]:.1f}\%$ & ${sub_acc[2]:.1f}\%$ & ${sub_acc[3]:.1f}\%$ \\\\' + '\n'
table += table_line

# plot results in same figure
rc('font',**{'family':'serif','serif':['Times']})

fig, ax = plt.subplots(1, 1, figsize = (8, 4))
ax.set_xlabel('Number of samples per class', fontsize = 14)
ax.set_ylabel('Test accuracy (%)', fontsize = 14)
# increase tick size
ax.tick_params(axis='both', which='major', labelsize=13)
# set y axis limits
if optenc:
    ax.set_ylim(15, 80)
else:
    ax.set_ylim(15, 75)
# make x axis logarithmic
ax.set_xscale('log')

method_names = {'GNN': ', MPNN', 'COCOA': ''}
methods = ['COCOA', 'GNN']
losses = ['COCOA', 'contrastive', 'time_loss', 'scratch']


learning_rates = {'GNN': 5e-4, 'COCOA': 5e-4}
loss_names = {'scratch': 'Scratch', 'contrastive': 'NT-Xent', 'time_loss': 'TS2Vec', 'COCOA': 'COCOA'}
# define 8 different markers for each method and loss
colors = {
    'GNN': ['salmon','orangered', 'red', 'coral'],
    'COCOA': ['blue','deepskyblue', 'lightblue',  'steelblue']
}
orders = [7, 5, 3, 1, 8, 6, 4, 2]

markers = ['o', 'v', 's', 'p', '*', 'h', 'D', 'X']
# format test accuarcy as latex results
i = 0
j = 0
for loss in losses:
    k = 0
    for method in methods:
        print(method, loss)
        # sort test accuracy by n_samples
        idx = np.argsort(all_results[method][learning_rates[method]][loss]['n_samples'])
        test_accuracy = np.array(all_results[method][learning_rates[method]][loss]['test_acc'])[idx]
        n_samples = np.array(all_results[method][learning_rates[method]][loss]['n_samples'])[idx]
        # plot with marker on top of line
        ax.plot(n_samples/5, test_accuracy*100, label = f'{loss_names[loss]}{method_names[method]}', 
                marker = markers[i], markersize = 6, color = colors[method][j], linewidth = 3, 
                zorder = orders[i]*10, alpha = 0.8)
        i+=1
        k+=1
    j+=1

ax.legend(loc = 'upper center', ncol = 4, fontsize = 13, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

table = ''
method_names = {'GNN': 'W. MPNN', 'COCOA': 'Wo. MPNN'}

## print results in overleaf table
for method in methods:
     for loss in losses:
        idx = np.argsort(all_results[method][learning_rates[method]][loss]['n_samples'])

        test_accuracy = np.array(all_results[method][learning_rates[method]][loss]['test_acc'])[idx]
        test_accuracy = np.reshape(test_accuracy, (6,5)).mean(1)
        n_samples = np.array(all_results[method][learning_rates[method]][loss]['n_samples'])[idx].reshape(6,5).mean(1)
        print(n_samples)
        sub_idx = [0,1,2,-1]
        sub_acc = test_accuracy[sub_idx]
        method_name = method_names[method]
        acc_str = [f'{s_acc:.3f}'[1:] for s_acc in sub_acc]
        table_line = f'{method_name} & {loss_names[loss]} & ${acc_str[0]}$ & ${acc_str[1]}$ & ${acc_str[2]}$ & ${acc_str[3]}$ \\\\' + '\n'
        table += table_line

print(table)