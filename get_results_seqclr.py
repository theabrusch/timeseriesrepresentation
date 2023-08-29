import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rc
# Set font family to Times New Roman

api = wandb.Api()

optenc = True
group_top = 'SeqCLR_ft'
learning_rates = [5e-4, 1e-3, 3e-3]
losses = ['contrastive']
all_results = dict()

for loss in losses:
    group = f'{group_top}_{loss}'
    cond = {"$and": [{"group": group}, {'config.ft_learning_rate': {'$in': learning_rates}}, {'config.optimize_encoder': optenc}, {'state': 'finished'}, {'config.suffix': '_R'}]}
    runs = api.runs(path = 'theabrusch/MultiView', filters = cond)
    all_results[loss] = dict()
    n_runs = len(runs)
    for run in runs:
        if not 'suffix' in run.config.keys():
            run.config['suffix'] = ''
        if 'seed' in run.config.keys():
            hist = run.history()
            map_nan = lambda x: np.nan if x == 'NaN' else x
            best_val_loss_arg = np.nanargmin(hist['val_class_loss'].apply(map_nan))

            best_val_acc = hist['val_acc'][best_val_loss_arg]
            best_test_acc = hist['test_acc'][best_val_loss_arg]
            n_samples = run.config['Finetune validation samples']
            lr = run.config['ft_learning_rate']
            suff = run.config['suffix']
            if lr in all_results[loss].keys():
                if suff in all_results[loss][lr].keys():
                    if n_samples in all_results[loss][lr][suff].keys():
                        all_results[loss][lr][suff][n_samples]['val_acc'].append(best_val_acc)
                        all_results[loss][lr][suff][n_samples]['test_acc'].append(best_test_acc)
                    else:
                        all_results[loss][lr][suff][n_samples] = dict()
                        all_results[loss][lr][suff][n_samples]['val_acc'] = [best_val_acc]
                        all_results[loss][lr][suff][n_samples]['test_acc'] = [best_test_acc]
                else:
                    all_results[loss][lr][suff]= dict()
                    all_results[loss][lr][suff][n_samples] = dict()
                    all_results[loss][lr][suff][n_samples]['val_acc'] = [best_val_acc]
                    all_results[loss][lr][suff][n_samples]['test_acc'] = [best_test_acc]
            else:
                all_results[loss][lr] = dict()
                all_results[loss][lr][suff] = dict()
                all_results[loss][lr][suff][n_samples] = dict()
                all_results[loss][lr][suff][n_samples]['val_acc'] = [best_val_acc]
                all_results[loss][lr][suff][n_samples]['test_acc'] = [best_test_acc]
                all_results[loss][lr][suff][n_samples]['n_samples'] = [n_samples]

best_lr = {50: 0, 250: 0, 500: 0, 5000: 0}
best_loss = {50: 0, 250: 0, 500: 0, 5000: 0}
best_suff = {50: 0, 250: 0, 500: 0, 5000: 0}
best_val_acc = {50: 0, 250: 0, 500: 0, 5000: 0}
best_test_acc = {50: 0, 250: 0, 500: 0, 5000: 0}


for loss in losses:
    for lr in learning_rates:
        for suff in all_results[loss][lr].keys():
            for n_samples in all_results[loss][lr][suff].keys():
                if np.mean(all_results[loss][lr][suff][n_samples]['val_acc']) > best_val_acc[n_samples]:
                    best_val_acc[n_samples] = np.mean(all_results[loss][lr][suff][n_samples]['val_acc'])
                    best_test_acc[n_samples] = np.mean(all_results[loss][lr][suff][n_samples]['test_acc'])
                    best_lr[n_samples] = lr
                    best_loss[n_samples] = loss
                    best_suff[n_samples] = suff
