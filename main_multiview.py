import torch
import argparse
from utils.ts2vec import TS2VecEncoder, TS2VecClassifer
from utils.multiview import GNNMultiview
from utils.dataset import get_datasets
from eegdataset import construct_eeg_datasets
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
    assert args.multi_channel_setup in ['None', 'sample_channel', 'avg_ch']
    print(mul_channel_explanations[args.multi_channel_setup])
    if not args.multi_channel_setup == 'sample_channel' and args.encoder == 'wave2vec' and args.pretrain:
        raise ValueError('Wave2Vec encoder is only available for multi-channel setup.')

    if 'eeg' in args.data_path or 'edf' in args.data_path:
         dset = args.data_path.split('/')[-1].strip('.yml')
    else:
        dset = args.data_path.split('/')[-2]
    #device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f'{args.output_path}/MultiView_{dset}_pretrain_{args.pretrain}_pretrain_subjs_{args.sample_pretrain_subjects}_multi_channel_setup_{args.multi_channel_setup}'
    
    output_path = check_output_path(output_path)
    args.outputh_path = output_path
    wandb.init(project = 'MultiView', config = args)
        
    print('Saving outputs in', output_path)

    if 'eeg' in args.data_path:
        args.train_mode = 'pretrain' if args.pretrain and not args.finetune else 'finetune' if args.finetune and not args.pretrain else 'both'
        args.standardize_epochs = 'channelwise'
        pretrain_loader, pretrain_val_loader, finetune_loader, finetune_val_loader, test_loader, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    else:
        args.ssl_mode = 'TS2Vec'
        args.downsample = False
        args.sample_channel = False
        pretrain_loader, pretrain_val_loader, test_loader, (channels, time_length, num_classes) = get_datasets(**vars(args))
        finetune_loader = pretrain_loader
        finetune_val_loader = pretrain_val_loader
    
    orig_channels = channels
    if args.multi_channel_setup == 'sample_channel' or args.multi_channel_setup == 'avg_ch':
        orig_channels = channels
        channels = 1
        
    if device.type == 'mps':
        norm = 'batch'
    else:
        norm = 'group'
    
    print('time', time_length, 'num classes', num_classes)

    model = GNNMultiview(channels = channels, time_length = time_length, num_classes = num_classes, norm = norm, **vars(args))
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
        model.fit(pretrain_loader,
                pretrain_val_loader,
                args.pretrain_epochs,
                optimizer,
                device,
                time_loss = not args.flatten,
                temperature = 0.5,
                log = True)
        
        time2 = datetime.datetime.now()    
        print('Pre-training for', args.pretrain_epochs,'epochs took', time2-time, 's.')

        if args.save_model:
            model.eval()
            path = f'{output_path}/pretrained_model.pt'
            torch.save(model.state_dict(), path)
    #if args.evaluate_latent_space:
    #    print('Evaluating latent space')
    #    train_outputs = model.evaluate_latent_space(finetune_loader, device=device, maxpool=True)
    #   val_outputs = model.evaluate_latent_space(finetune_val_loader, device=device, maxpool=True)
    #    test_outputs = model.evaluate_latent_space(test_loader, device=device, maxpool=True)

    #    with open(f'{output_path}/pretrain_latents.pickle', 'wb') as file:
    #            pickle.dump(train_outputs, file)
    #    with open(f'{output_path}/pretrain_val_latents.pickle', 'wb') as file:
    #            pickle.dump(val_outputs, file)
    #    with open(f'{output_path}/pretrain_test_latents.pickle', 'wb') as file:
    #            pickle.dump(test_outputs, file)

    if args.finetune:
        print('Finetuning model')

        wandb.config.update({'Finetune samples': len(finetune_loader.dataset), 'Finetune validation samples': len(finetune_val_loader.dataset), 'Test samples': len(test_loader.dataset)})

        if args.optimize_encoder:
            optimizer = AdamW(model.parameters(), lr = args.ft_learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = AdamW(model.classifier.parameters(), lr = args.ft_learning_rate, weight_decay=args.weight_decay)

        
        if 'eeg' in args.data_path:
            targets = finetune_loader.dataset.dn3_dset.get_targets()
        else:
            targets = finetune_loader.dataset.Y

        if not args.balanced_sampling == 'finetune' or args.balanced_sampling == 'both':
            weights = torch.tensor(compute_class_weight('balanced', classes = np.unique(targets), y = targets)).float().to(device)
        else:
            weights = None
        
        wandb.config.update({'Target distribution': np.unique(targets, return_counts=True)[-1]})

        model.finetune(
                 finetune_loader,
                 finetune_val_loader,
                 args.finetune_epochs,
                 optimizer,
                 weights,
                 device,
                 choose_best = args.choose_best,
        )

        accuracy, prec, rec, f = model.evaluate_classifier(test_loader, device)
        wandb.config.update({'Test accuracy': accuracy, 'Test precision': prec, 'Test recall': rec, 'Test f1': f})
        print('test accuracy', accuracy)
        print('test precision', prec)
        print('test recall', rec)
        print('test f1', f)

        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--save_model', type = eval, default = False)
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--evaluate_latent_space', type = eval, default = False)
    parser.add_argument('--finetune', type = eval, default = True)
    parser.add_argument('--optimize_encoder', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)
    parser.add_argument('--output_path', type = str, default = 'outputs')

    # data arguments
    parser.add_argument('--data_path', type = str, default = 'sleepeeg_local.yml')
    parser.add_argument('--finetune_path', type = str, default = 'same')
    parser.add_argument('--balanced_sampling', type = str, default = 'finetune')
    parser.add_argument('--seed_generator', type = eval, default = True)

    # model arguments
    parser.add_argument('--flatten', type = eval, default = False)
    parser.add_argument('--pool', type = str, default = 'adapt_avg')
    parser.add_argument('--encoder', type = str, default = 'wave2vec')
    parser.add_argument('--layers', type = int, default = 6)
    parser.add_argument('--choose_best', type = eval, default = True)
    parser.add_argument('--conv_do', type = float, default = 0.1)
    parser.add_argument('--feat_do', type = float, default = 0.1)
    parser.add_argument('--num_message_passing_rounds', type = int, default = 3)
    parser.add_argument('--hidden_channels', type = int, default = 256)
    parser.add_argument('--out_dim', type = int, default = 64)


    # eeg arguments
    parser.add_argument('--sample_pretrain_subjects', type = eval, default = 3)
    parser.add_argument('--sample_finetune_train_subjects', type = eval, default = 1)
    parser.add_argument('--sample_finetune_val_subjects', type = eval, default = 1)
    parser.add_argument('--sample_test_subjects', type = eval, default = 1)

    # augmentation arguments
    parser.add_argument('--multi_channel_setup', type = str, default = 'sample_channel') # None, sample_channel, ch_avg

    # optimizer arguments
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--ft_learning_rate', type = float, default = 1e-3)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--pretrain_epochs', type = int, default = 10)
    parser.add_argument('--finetune_epochs', type = int, default = 1)
    parser.add_argument('--batchsize', type = int, default = 128)
    parser.add_argument('--target_batchsize', type = int, default = 128)
    args = parser.parse_args()
    main(args)

