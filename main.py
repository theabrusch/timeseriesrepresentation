import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from utils.trainer import TFC_trainer, evaluate_latent_space, finetune_model, evaluate_model
from utils.ts2vec import TS2VecEncoder, TS2VecClassifer
from utils.dataset import TFC_Dataset, get_datasets, get_dset_info
from eegdataset import construct_eeg_datasets
from torch.optim import AdamW
from utils.plot_functions import plot_contrastive_losses
import pickle
import os
import datetime
from prettytable import PrettyTable
import wandb

def params_to_tb(writer, args):
    t = PrettyTable(['Argument', 'Value'])
    param_dict = vars(args)
    for key, val in param_dict.items():
        t.add_row([key, val])
    writer.add_text("args", t.get_html_string(), global_step=0)

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

def results_to_tb(results, writer, dset):
    t = PrettyTable(['Argument', 'Value'])
    for key, val in results.items():
        t.add_row([key, val])
    writer.add_text(f"{dset}_results", t.get_html_string(), global_step=0)
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
    
    if not args.overwrite:
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
                                                                                                                                                            standardize_epochs = True,
                                                                                                                                                            balanced_sampling= 'finetune',
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
         channels = 1
    model = TS2VecEncoder(input_size=channels, avg_channels = args.multi_channel_setup=='avg_ch', hidden_channels=64, out_dim=320)
    model.to(device)
    if args.load_model:
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))

    if args.pretrain:
        # get datasets
        print('Initializing model')
        
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
        classifier = TS2VecClassifer(in_features=320, n_classes=num_classes, pool = args.pool)
        classifier.to(device)
        if args.optimize_encoder:
            optimizer = AdamW(list(model.parameters())+list(classifier.parameters()), lr = args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = AdamW(list(classifier.parameters()), lr = args.learning_rate, weight_decay=args.weight_decay)
        
        classifier = model.finetune(
            finetune_loader,
            finetune_val_loader, 
            classifier,
            optimizer,
            args.finetune_epochs,
            device,
            log = True,
            choose_best = args.choose_best
        )

        accuracy, prec, rec, f = model.evaluate_classifier(test_loader, classifier, device)
        print('test accuracy', accuracy)
        print('test precision', prec)
        print('test recall', rec)
        print('test f1', f)

        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--save_model', type = eval, default = True)
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--evaluate_latent_space', type = eval, default = True)
    parser.add_argument('--finetune', type = eval, default = True)
    parser.add_argument('--optimize_encoder', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)

    # data arguments
    parser.add_argument('--data_path', type = str, default = 'datasets/HAR/')
    parser.add_argument('--finetune_path', type = str, default = 'same')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--target_batch_size', type = int, default = 22)
    parser.add_argument('--output_path', type = str, default = 'outputs')
    parser.add_argument('--overwrite', type = eval, default = False)

    # model arguments
    parser.add_argument('--pool', type = str, default = 'max')
    parser.add_argument('--choose_best', type = eval, default = True)

    # eeg arguments
    parser.add_argument('--sample_pretrain_subjs', type = eval, default = 3)
    parser.add_argument('--sample_finetune_train_subjs', type = eval, default = 1)
    parser.add_argument('--sample_finetune_val_subjs', type = eval, default = 1)
    parser.add_argument('--sample_test_subjs', type = eval, default = 2)

    # augmentation arguments
    parser.add_argument('--multi_channel_setup', type = str, default = 'sample_channel') # None, sample_channel, ch_avg

    # optimizer arguments
    parser.add_argument('--temporal_unit', type = int, default = 2)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--alpha', type = float, default=0.5)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--finetune_epochs', type = int, default = 1)
    parser.add_argument('--epochs', type = int, default = 0)
    args = parser.parse_args()
    main(args)

