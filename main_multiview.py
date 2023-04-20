import torch
import argparse
from utils.multiview import load_model, pretrain, finetune, evaluate_classifier
from eegdataset import construct_eeg_datasets
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import wandb

def check_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        i = 1
        while os.path.exists(output_path + f'_v_{i}'):
            i+=1
        output_path = output_path + f'_v_{i}'
        os.makedirs(output_path, exist_ok=True)
    return output_path

mul_channel_explanations = {
     'None': 'Multi channel setup is set to None.',
     'sample_channel': 'Multi channel setup is set to sample_channel. This means that sampled channels will be used as each others augmented versions.',
     'avg_ch': 'Multi channel setup is set to ch_avg. This means that the channels are averaged before convolutions.'
}
def main(args):
    assert args.multi_channel_setup in ['None', 'sample_channel', 'avg_ch']
    print(mul_channel_explanations[args.multi_channel_setup])
    if not args.multi_channel_setup == 'sample_channel' and args.encoder == 'wave2vec' and args.pretrain:
        raise ValueError('Wave2Vec encoder is only available for multi-channel setup.')

    dset = args.data_path.split('/')[-1].strip('.yml')
    output_path = f'{args.output_path}/MultiView_{dset}_pretrain_{args.pretrain}_pretrain_subjs_{args.sample_pretrain_subjects}_multi_channel_setup_{args.multi_channel_setup}'
    
    output_path = check_output_path(output_path)
    args.outputh_path = output_path
        
    print('Saving outputs in', output_path)

    args.train_mode = 'pretrain' if args.pretrain and not args.finetune else 'finetune' if args.finetune and not args.pretrain else 'both'
    args.standardize_epochs = 'channelwise'
    pretrain_loader, pretrain_val_loader, finetune_loader, finetune_val_loader, test_loader, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    
    orig_channels = channels
    if args.multi_channel_setup == 'sample_channel' or args.multi_channel_setup == 'avg_ch':
        orig_channels = channels
        channels = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pretrain:
        wandb.init(project = 'MultiView', group = args.pretraining_setup, config = args)
        model, loss_fn = load_model(args.pretraining_setup, device, channels, time_length, num_classes, args)

        if args.load_model:
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))

        wandb.config.update({'Pretrain samples': len(pretrain_loader.dataset), 'Pretrain validation samples': len(pretrain_val_loader.dataset)})
        
        optimizer = AdamW(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        # pretrain model
        pretrain(model,
                pretrain_loader,
                pretrain_val_loader,
                args.pretrain_epochs,
                optimizer,
                device,
                backup_path=output_path,
                loss_fn = loss_fn,
                log = True)

        if args.save_model:
            model.eval()
            path = f'{output_path}/pretrained_model.pt'
            torch.save(model.state_dict(), path)
        wandb.finish()

    if args.finetune:

        for ft_loader, ft_val_loader in zip(finetune_loader, finetune_val_loader):
            wandb.init(project = 'MultiView', group = args.pretraining_setup, config = args)
            model, loss_fn = load_model(args.pretraining_setup, device, channels, time_length, num_classes, args)

            if args.load_model:
                model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))

            wandb.config.update({'Finetune samples': len(ft_loader.dataset), 'Finetune validation samples': len(ft_val_loader.dataset), 'Test samples': len(test_loader.dataset)})

            if args.optimize_encoder:
                optimizer = AdamW(model.parameters(), lr = args.ft_learning_rate, weight_decay=args.weight_decay)
            else:
                optimizer = AdamW(model.classifier.parameters(), lr = args.ft_learning_rate, weight_decay=args.weight_decay)
            if args.pretraining_setup != 'GNN':
                model.update_classifier(num_classes, orig_channels=orig_channels)
                model.to(device)

            targets = ft_loader.dataset.dn3_dset.get_targets()
            if not args.balanced_sampling == 'finetune' or args.balanced_sampling == 'both':
                weights = torch.tensor(compute_class_weight('balanced', classes = np.unique(targets), y = targets)).float().to(device)
            else:
                weights = None
            
            wandb.config.update({'Target distribution': np.unique(targets, return_counts=True)[-1]})

            finetune(model,
                    ft_loader,
                    ft_val_loader,
                    args.finetune_epochs,
                    optimizer,
                    weights,
                    device,
                    test_loader = test_loader if args.track_test_performance else None,
                    early_stopping_criterion=args.early_stopping_criterion,
            )

            accuracy, prec, rec, f = evaluate_classifier(model, test_loader, device)
            wandb.config.update({'Test accuracy': accuracy, 'Test precision': prec, 'Test recall': rec, 'Test f1': f})
            wandb.finish()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--job_id', type = str, default = '0')
    parser.add_argument('--save_model', type = eval, default = False)
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--evaluate_latent_space', type = eval, default = False)
    parser.add_argument('--finetune', type = eval, default = True)
    parser.add_argument('--optimize_encoder', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)
    parser.add_argument('--output_path', type = str, default = 'outputs')
    parser.add_argument('--pretraining_setup', type = str, default = 'COCOA')

    # data arguments
    parser.add_argument('--data_path', type = str, default = 'sleepeeg_local.yml')
    parser.add_argument('--finetune_path', type = str, default = 'sleepedf_local.yml')
    parser.add_argument('--balanced_sampling', type = str, default = 'finetune')
    parser.add_argument('--seed_generator', type = eval, default = '[10,20]')

    # model arguments
    parser.add_argument('--pool', type = str, default = 'adapt_avg')
    parser.add_argument('--encoder', type = str, default = 'wave2vec')
    parser.add_argument('--layers', type = int, default = 6)
    parser.add_argument('--early_stopping_criterion', type = str, default = None)
    parser.add_argument('--conv_do', type = float, default = 0.1)
    parser.add_argument('--feat_do', type = float, default = 0.1)
    parser.add_argument('--num_message_passing_rounds', type = int, default = 3)
    parser.add_argument('--hidden_channels', type = int, default = 256)
    parser.add_argument('--out_dim', type = int, default = 64)


    # eeg arguments
    parser.add_argument('--sample_pretrain_subjects', type = eval, default = 3)
    parser.add_argument('--sample_finetune_train_subjects', type = eval, default = 3)
    parser.add_argument('--sample_finetune_val_subjects', type = eval, default = 1)
    parser.add_argument('--sample_test_subjects', type = eval, default = 1)

    # augmentation arguments
    parser.add_argument('--multi_channel_setup', type = str, default = 'sample_channel') # None, sample_channel, ch_avg

    # optimizer arguments
    parser.add_argument('--loss', type = str, default = 'time_loss')
    parser.add_argument('--track_test_performance', type = eval, default = True)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--ft_learning_rate', type = float, default = 1e-3)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--pretrain_epochs', type = int, default = 10)
    parser.add_argument('--finetune_epochs', type = int, default = 1)
    parser.add_argument('--batchsize', type = int, default = 128)
    parser.add_argument('--target_batchsize', type = int, default = 128)
    args = parser.parse_args()
    main(args)

