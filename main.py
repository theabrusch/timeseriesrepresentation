import torch
import argparse
from utils.ts2vec import TS2VecEncoder, TS2VecClassifer
from utils.dataset import get_datasets
from src.eegdataset import construct_eeg_datasets
from torch.optim import AdamW
from utils.plot_functions import plot_contrastive_losses
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import datetime
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
    if 'eeg' in args.data_path:
         dset = args.data_path.split('/')[-1].strip('.yml')
    else:
        dset = args.data_path.split('/')[-2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f'{args.output_path}/ts2vec_{dset}'

    assert args.multi_channel_setup in ['None', 'sample_channel', 'avg_ch']
    print(mul_channel_explanations[args.multi_channel_setup])

    if not args.multi_channel_setup == 'sample_channel' and args.encoder == 'wave2vec' and args.pretrain:
        raise ValueError('Wave2Vec encoder is only available for multi-channel setup.')
    
    output_path = check_output_path(output_path)
    args.outputh_path = output_path
    wandb.init(project = 'ts2vec', config = args)
        
    print('Saving outputs in', output_path)

    if 'eeg' in args.data_path:
        train_mode = 'both'
        pretrain_loader, pretrain_val_loader, finetune_loader, finetune_val_loader, test_loader, (channels, time_length, num_classes) = construct_eeg_datasets(args.data_path, 
                                                                                                                                                            args.finetune_path, 
                                                                                                                                                            batchsize = args.batch_size,
                                                                                                                                                            normalize = False, 
                                                                                                                                                            standardize_epochs = 'channelwise',
                                                                                                                                                            balanced_sampling= args.balanced_sampling,
                                                                                                                                                            target_batchsize = args.target_batch_size,
                                                                                                                                                            sample_pretrain_subjects = args.sample_pretrain_subjs,
                                                                                                                                                            sample_finetune_train_subjects = args.sample_finetune_train_subjs,
                                                                                                                                                            sample_finetune_val_subjects = args.sample_finetune_val_subjs,
                                                                                                                                                            sample_test_subjects = args.sample_test_subjs,
                                                                                                                                                            train_mode = train_mode)
    else:
         pretrain_loader, pretrain_val_loader, test_loader, (channels, time_length, num_classes) = get_datasets(data_path = args.data_path,
                                                                                                                ssl_mode='TS2Vec',
                                                                                                                downsample=False,
                                                                                                                sample_channel = False, 
                                                                                                                batch_size=args.batch_size)
    if args.multi_channel_setup == 'sample_channel' or args.multi_channel_setup == 'avg_ch':
        orig_channels = channels
        channels = 1
    else:
        orig_channels = channels
          
    model = TS2VecEncoder(input_size=channels, avg_channels = args.multi_channel_setup=='avg_ch', encoder = args.encoder, hidden_channels=64, out_dim=320)
    model.to(device)
    if args.load_model:
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))

    if args.pretrain:
        # get datasets
        print('Initializing model')
        wandb.config.update({'Pretrain samples': len(pretrain_loader.dataset), 'Pretrain validation samples': len(pretrain_val_loader.dataset)})
        
        optimizer = AdamW(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        print('Training model')
        time = datetime.datetime.now()
        # pretrain model
        losses = model.fit(dataloader = pretrain_loader,
                            val_dataloader = pretrain_val_loader,
                            epochs = args.epochs,
                            optimizer = optimizer,
                            alpha = args.alpha,
                            augmentation_type = 'channels' if args.multi_channel_setup == 'sample_channel' else 'crop',
                            temporal_unit = args.temporal_unit,
                            backup_path = output_path,
                            device = device)
        time2 = datetime.datetime.now()    
        print('Pre-training for', args.epochs,'epochs took', time2-time, 's.')

        if args.save_model:
            model.eval()
            path = f'{output_path}/pretrained_model.pt'
            torch.save(model.state_dict(), path)
        plot_contrastive_losses(losses['train'], f'{output_path}/pretrain_train_losses.png')
        plot_contrastive_losses(losses['val'], f'{output_path}/pretrain_val_losses.png')
        with open(f'{output_path}/pretrain_losses.pickle', 'wb') as file:
                pickle.dump(losses, file)
    if args.evaluate_latent_space:
        print('Evaluating latent space')
        if 'eeg' in args.data_path:
            train_outputs = model.evaluate_latent_space(finetune_loader, device=device, maxpool=True)
            val_outputs = model.evaluate_latent_space(finetune_val_loader, device=device, maxpool=True)
        else:
            train_outputs = model.evaluate_latent_space(pretrain_loader, device=device, maxpool=True)
            val_outputs = model.evaluate_latent_space(pretrain_val_loader, device=device, maxpool=True)
        test_outputs = model.evaluate_latent_space(test_loader, device=device, maxpool=True)

        with open(f'{output_path}/pretrain_latents.pickle', 'wb') as file:
                pickle.dump(train_outputs, file)
        with open(f'{output_path}/pretrain_val_latents.pickle', 'wb') as file:
                pickle.dump(val_outputs, file)
        with open(f'{output_path}/pretrain_test_latents.pickle', 'wb') as file:
                pickle.dump(test_outputs, file)
    if args.finetune:
        if 'HAR' in args.data_path:
             finetune_loader = pretrain_loader
             finetune_val_loader = pretrain_val_loader
        classifier = TS2VecClassifer(in_features=320, n_classes=num_classes, pool = args.pool, orig_channels = orig_channels)
        classifier.to(device)

        wandb.config.update({'Finetune samples': len(finetune_loader.dataset), 'Finetune validation samples': len(finetune_val_loader.dataset), 'Test samples': len(test_loader.dataset)})

        if args.optimize_encoder:
            optimizer = AdamW(list(model.parameters())+list(classifier.parameters()), lr = args.ft_learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = AdamW(list(classifier.parameters()), lr = args.ft_learning_rate, weight_decay=args.weight_decay)

        if 'eeg' in args.data_path:
            targets = finetune_loader.dataset.dn3_dset.get_targets()
            weights = torch.tensor(compute_class_weight('balanced', classes = np.unique(targets), y = targets)).float()
        else:
            targets = finetune_loader.dataset.Y
            weights = torch.tensor(compute_class_weight('balanced', classes = np.unique(targets), y = targets)).float()
        
        wandb.config.update({'Target distribution': np.unique(targets, return_counts=True)[-1]})

        classifier = model.finetune(
            finetune_loader,
            finetune_val_loader, 
            classifier,
            optimizer,
            args.finetune_epochs,
            device,
            weights = weights,
            log = True,
            choose_best = args.choose_best
        )

        accuracy, prec, rec, f = model.evaluate_classifier(test_loader, classifier, device)
        wandb.config.update({'Test accuracy': accuracy, 'Test precision': prec, 'Test recall': rec, 'Test f1': f})
        print('test accuracy', accuracy)
        print('test precision', prec)
        print('test recall', rec)
        print('test f1', f)

        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--save_model', type = eval, default = True)
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--pretrain', type = eval, default = False)
    parser.add_argument('--evaluate_latent_space', type = eval, default = False)
    parser.add_argument('--finetune', type = eval, default = True)
    parser.add_argument('--optimize_encoder', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)

    # data arguments
    parser.add_argument('--data_path', type = str, default = 'sleepeeg_local.yml')
    parser.add_argument('--finetune_path', type = str, default = 'sleepedf_local.yml')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--target_batch_size', type = int, default = 128)
    parser.add_argument('--output_path', type = str, default = 'outputs')
    parser.add_argument('--balanced_sampling', type = str, default = 'finetune')

    # model arguments
    parser.add_argument('--pool', type = str, default = 'adapt_avg')
    parser.add_argument('--encoder', type = str, default = 'wave2vec')
    parser.add_argument('--choose_best', type = eval, default = True)

    # eeg arguments
    parser.add_argument('--sample_pretrain_subjs', type = eval, default = 3)
    parser.add_argument('--sample_finetune_train_subjs', type = eval, default = 3)
    parser.add_argument('--sample_finetune_val_subjs', type = eval, default = 3)
    parser.add_argument('--sample_test_subjs', type = eval, default = 3)

    # augmentation arguments
    parser.add_argument('--multi_channel_setup', type = str, default = 'None') # None, sample_channel, ch_avg

    # optimizer arguments
    parser.add_argument('--temporal_unit', type = int, default = 2)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--ft_learning_rate', type = float, default = 1e-3)
    parser.add_argument('--alpha', type = float, default=0.5)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--finetune_epochs', type = int, default = 1)
    parser.add_argument('--epochs', type = int, default = 0)
    args = parser.parse_args()
    main(args)

