import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from utils.trainer import TFC_trainer, evaluate_latent_space, finetune_model, evaluate_model
from utils.ts2vec import TS2VecEncoder
from utils.dataset import TFC_Dataset, get_datasets, get_dset_info
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

def main(args):
    wandb.init(project = 'ts2vec', config = args)
    dset = args.data_path.split('/')[-2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f'{args.output_path}/ts2vec_{dset}'

    if not args.overwrite:
        output_path = check_output_path(output_path)
        
    print('Saving outputs in', output_path)

    if args.pretrain:
        # get datasets
        train_loader, val_loader, test_loader, (channels, time_length, num_classes) = get_datasets(data_path = args.data_path, 
                                                                                                    ssl_mode='TS2Vec',
                                                                                                    sample_channel = args.sample_channel, 
                                                                                                    batch_size=args.batch_size)
        
        print('Initializing model')
        model = TS2VecEncoder(input_size=channels, hidden_channels=64, out_dim=320)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        print('Training model')
        time = datetime.datetime.now()
        # pretrain model
        losses = model.fit(dataloader = train_loader,
                            val_dataloader = val_loader,
                            epochs = args.epochs,
                            optimizer = optimizer,
                            alpha = args.alpha,
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

        print('Evaluating latent space')
        train_outputs = model.evaluate_latent_space(train_loader, device=device, maxpool=True)
        val_outputs = model.evaluate_latent_space(val_loader, device=device, maxpool=True)
        test_outputs = model.evaluate_latent_space(test_loader, device=device, maxpool=True)

        with open(f'{output_path}/pretrain_latents.pickle', 'wb') as file:
                pickle.dump(train_outputs, file)
        with open(f'{output_path}/pretrain_val_latents.pickle', 'wb') as file:
                pickle.dump(val_outputs, file)
        with open(f'{output_path}/pretrain_test_latents.pickle', 'wb') as file:
                pickle.dump(test_outputs, file)


        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--save_model', type = eval, default = True)
    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)
    # data arguments
    parser.add_argument('--data_path', type = str, default = 'datasets/HAR/')
    parser.add_argument('--finetune_path', type = str, default = 'datasets/Gesture/')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--target_batch_size', type = int, default = 22)
    parser.add_argument('--output_path', type = str, default = 'outputs')
    parser.add_argument('--overwrite', type = eval, default = False)

    # augmentation arguments
    parser.add_argument('--sample_channel', type = eval, default = False)
    # optimizer arguments
    parser.add_argument('--learning_rate', type = float, default = 3e-6)
    parser.add_argument('--alpha', type = float, default=0.5)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--epochs', type = int, default = 0 )
    args = parser.parse_args()
    main(args)

