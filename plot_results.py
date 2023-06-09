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
readout = True

all_results = dict()
for optenc in [True, False]:
    all_results[str(optenc)] = dict()
    for method in methods:
        all_results[str(optenc)][method] = dict()
        for lr in learning_rates[method]:
            all_results[str(optenc)][method][lr] = dict()
            for loss in losses:
                all_results[str(optenc)][method][lr][loss] = dict()
                all_results[str(optenc)][method][lr][loss]['val_acc'] = []
                all_results[str(optenc)][method][lr][loss]['test_acc'] = []
                all_results[str(optenc)][method][lr][loss]['n_samples'] = []
                group = f'{method}_{loss}_readout' if readout else f'{method}_{loss}'
                if method == 'GNN' and optenc:
                    cond = {"$and": [{"group": group}, {'config.ft_learning_rate': lr}, {'config.optimize_encoder': optenc}, {'config.feat_do':0.1}]}
                elif not optenc and method == 'GNN' and loss == 'time_loss':
                    cond = {"$and": [{"group": group}, {'config.ft_learning_rate': lr}, {'config.optimize_encoder': optenc}]}
                elif not optenc:
                    cond = {"$and": [{"group": group}, {'config.ft_learning_rate': lr}, {'config.optimize_encoder': optenc}, {'config.feat_do':0.4}]}
                else:
                    cond = {"$and": [{"group": group}, {'config.ft_learning_rate': lr}, {'config.optimize_encoder': optenc}]}
                runs = api.runs(path = 'theabrusch/MultiView', filters = cond)
                print(f'{method}_{loss}')
                print(f'runs: {len(runs)}\n')
                for run in runs:
                    #feat_do = 0.4
                    #if run.group == 'GNN_time_loss_readout' and run.config['Finetune samples'] in [2500,5000] and run.config['seed'] == 441:
                    #    feat_do=0.1
                    if 'seed' in run.config.keys():
                        hist = run.history()
                        best_val_loss_arg = np.argmin(hist['val_class_loss'])
                        best_val_acc = hist['val_acc'][best_val_loss_arg]
                        best_test_acc = hist['test_acc'][best_val_loss_arg]
                        n_samples = run.config['Finetune validation samples']
                        all_results[str(optenc)][method][lr][loss]['val_acc'].append(best_val_acc)
                        all_results[str(optenc)][method][lr][loss]['test_acc'].append(best_test_acc)
                        all_results[str(optenc)][method][lr][loss]['n_samples'].append(n_samples)

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

method_names = {'GNN': ', MPNN', 'COCOA': ''}
methods = ['COCOA', 'GNN']
losses = ['COCOA', 'contrastive', 'time_loss', 'scratch']
optenc_options = ['True', 'False']


learning_rates = {'GNN': 5e-4, 'COCOA': 5e-4}
loss_names = {'scratch': 'Scratch', 'contrastive': 'NT-Xent', 'time_loss': 'TS2Vec', 'COCOA': 'COCOA'}
# define 8 different markers for each method and loss
colors = {
    'GNN': ['blue','orangered', 'green',  'purple'],
    'COCOA': ['lightskyblue','lightsalmon', 'lightgreen', 'plum'],
}
orders = [7, 5, 3, 1, 8, 6, 4, 2]

markers = ['o', 'v', 's', 'p', '*', 'h', 'D', 'X']

rc('font',**{'family':'serif','serif':['Times']})

fig, axes = plt.subplots(2, 1, figsize = (9, 8))
axes[1].set_xlabel('Number of samples per class', fontsize = 16)
axes[0].set_ylabel('Test accuracy (%)', fontsize = 16)
axes[1].set_ylabel('Test accuracy (%)', fontsize = 16)
# increase tick size
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=14)
# set y axis limits
axes[0].set_ylim(20, 75)
axes[1].set_ylim(15, 75)
axes[0].set_xlim(9, 1050)
axes[1].set_xlim(9, 1050)
# make x axis logarithmic
axes[0].set_xscale('log')
axes[1].set_xscale('log')

# format test accuarcy as latex results
for optenc, ax in zip(optenc_options, axes):
    res = all_results[str(optenc)]
    i = 0
    j = 0
    for loss in losses:
        k = 0
        for method in methods:
            print(method, loss)
            # sort test accuracy by n_samples
            idx = np.argsort(res[method][learning_rates[method]][loss]['n_samples'])
            test_accuracy = np.array(res[method][learning_rates[method]][loss]['test_acc'])[idx].reshape(7,5).mean(1)
            n_samples = np.array(res[method][learning_rates[method]][loss]['n_samples'])[idx].reshape(7,5).mean(1)
            # plot with marker on top of line
            ax.plot(n_samples/5, test_accuracy*100, label = f'{loss_names[loss]}{method_names[method]}', 
                    marker = markers[i], markersize = 6, color = colors[method][j], linewidth = 3, 
                    zorder = orders[i]*10, alpha = 0.8)
            
            i+=1
            k+=1
        j+=1
    if optenc == 'True':
        ax.set_title(f'Optimize entire network', fontsize = 16)
    else:
        ax.set_title(f'Optimize only classifier', fontsize = 16)

axes[1].legend(loc = 'upper center', ncol = 4, fontsize = 14, bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

table1 = ''
table2 = ''
method_names = {'GNN': 'W. MPNN', 'COCOA': 'Wo. MPNN'}

## print results in overleaf table
for method in methods:
     for loss in losses:
        idx = np.argsort(all_results[method][learning_rates[method]][loss]['n_samples'])

        test_accuracy = np.array(all_results[method][learning_rates[method]][loss]['test_acc'])[idx]
        test_mean = np.reshape(test_accuracy, (7,5)).mean(1)
        test_std = np.reshape(test_accuracy, (7,5)).std(1)
        n_samples = np.array(all_results[method][learning_rates[method]][loss]['n_samples'])[idx].reshape(7,5).mean(1)
        
        print(method, loss)
        sub_idx = [0,2,3,-1]
        sub_acc = test_mean[sub_idx]
        method_name = method_names[method]
        acc_str = [f'{s_acc:.3f}'[1:] for s_acc in sub_acc]
        print(acc_str)
        std_str = [f'{s_std:.2f}'[1:] for s_std in test_std[sub_idx]]
        print(std_str)
        table_line = f'{method_name} & {loss_names[loss]} & ${acc_str[0]}$ & ${acc_str[1]}$ & ${acc_str[2]}$ & ${acc_str[3]}$ \\\\' + '\n'
        table1 += table_line
        table_line = f'{method_name} & {loss_names[loss]} & ${acc_str[0]}\pm{std_str[0]}$ & ${acc_str[1]}\pm{std_str[1]}$ & ${acc_str[2]}\pm{std_str[2]}$ & ${acc_str[3]}\pm{std_str[3]}$ \\\\' + '\n'
        table2 += table_line
print(table1)

print(table2)