import torch
import argparse
from torch.utils.data import DataLoader
from utils.trainer import TFC_trainer, evaluate_latent_space, finetune_model, evaluate_model
from utils.models import TFC_encoder, ContrastiveLoss, ContrastiveLoss2, ClassifierModule
from utils.dataset import TFC_Dataset, get_datasets, get_dset_info
from eegdataset import construct_eeg_datasets
from torch.optim import Adam
from utils.plot_functions import plot_contrastive_losses
import pickle
import os
import datetime
from prettytable import PrettyTable
#from torch.utils.tensorboard import SummaryWriter

def params_to_tb(writer, args):
    t = PrettyTable(['Argument', 'Value'])
    param_dict = vars(args)
    for key, val in param_dict.items():
        t.add_row([key, val])
    writer.add_text("args", t.get_html_string(), global_step=0)

def check_output_path(output_path):
    if not os.path.exists(output_path):
                os.makedirs(output_path)
    elif len(os.listdir(output_path)) == 0:
        output_path = output_path
    else:
        i = 1
        while os.path.exists(output_path + f'_v_{i}') and not len(os.listdir(output_path + f'_v_{i}')) == 0:
            i+=1
        output_path = output_path + f'_v_{i}'
        os.makedirs(output_path, exist_ok=True)
    return output_path


def main(args):
    dset = args.config_path.split('/')[-1].strip('.yml')
    if args.finetune_path == 'same':
        finetune_dset = dset
    else:
        finetune_dset = args.finetune_path.split('/')[-1].strip('.yml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f'{args.output_path}/TFC_{args.train_TFC}_multchannel_{args.avg_channels}_{dset}'

    # write to tensorboard
    #writer = SummaryWriter(f'../runs/TFC_pretrain_{dset}_finetune_{finetune_dset}_{str(datetime.now())}')
    #params_to_tb(writer, args)

    output_path = check_output_path(output_path)
        
    print('Saving outputs in', output_path)
    if args.avg_channels == 'after':
        args.sample_channel = True
        avg_channels_before = False
        avg_channels_after = True
    elif args.avg_channels == 'before':
        args.sample_channel = True
        avg_channels_before = True
        avg_channels_after = False
    elif args.avg_channels == 'None':
        args.sample_channel = False
        avg_channels_before = False
        avg_channels_after = False

    train_mode = 'both' if args.pretrain and args.finetune else 'finetune' if args.finetune else 'pretrain'
    pretrain_loader, pretrain_val_loader, finetune_loader, finetune_val_loader, test_loader, (channels, time_length, num_classes) = construct_eeg_datasets(args.config_path, 
                                                                                                                                                           args.finetune_path, 
                                                                                                                                                           batchsize = args.batch_size, 
                                                                                                                                                           target_batchsize = args.target_batch_size,
                                                                                                                                                           sample_subjects = args.sample_subjs,
                                                                                                                                                           sample_test_subjects = args.sample_test_subjs,
                                                                                                                                                           train_mode = train_mode)

    if args.pretrain:
        print('Initializing model')
        model = TFC_encoder(in_channels = channels, input_size = time_length, avg_channels_before = avg_channels_before, avg_channels_after=avg_channels_after,
                            num_classes = num_classes, stride = args.stride, classify = False)
        model.to(device)
        optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        loss_fn = ContrastiveLoss(tau = 0.2, device = device)

        print('Training model')
        time = datetime.datetime.now()
        # pretrain model
        model, losses = TFC_trainer(model = model, 
                                    train_loader = pretrain_loader, 
                                    optimizer = optimizer, 
                                    loss_fn = loss_fn, 
                                    epochs = args.epochs, 
                                    val_loader = pretrain_val_loader, 
                                    device = device, 
                                    classifier = None,
                                    class_optimizer = None)
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
    else:
        # load pretrained model
        if args.pretrained_model_path is not None:
            pretrained_path = args.pretrained_model_path
        else:
            pretrained_path = f'{output_path}/pretrained_model.pt'

        channels, time_length = get_dset_info(args.data_path, sample_channel = args.sample_channel)
        model = TFC_encoder(in_channels = channels, input_size = time_length, classify = False, 
                            num_classes = 6, stride = args.stride, avg_channels_before = avg_channels_before, 
                            avg_channels_after = avg_channels_after)

        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.to(device=device)
        if args.loss == 'poly':
            loss_fn = ContrastiveLoss(tau = 0.2, device = device)
        else:
            loss_fn = ContrastiveLoss2(tau = 0.2, device = device)

    if args.evaluate_latent_space:
            # evaluate and save latent space for pretraining dataset
            time = datetime.datetime.now() 
            _, val_loader, _, _ = get_datasets(data_path = args.data_path, 
                                                abs_budget=args.abs_budget, 
                                                batch_size=args.batch_size)

            outputs = evaluate_latent_space(model = model, data_loader = val_loader, device = device, classifier = args.train_classifier, save_h = False)

            time2 = datetime.datetime.now()   
            print('Evaluating the latent space took', time2-time, 's.')
            
            with open(f'{output_path}/pretrain_latent_variables.pickle', 'wb') as file:
                pickle.dump(outputs, file)

    if args.finetune:

        finetune_dset = dset
        # evaluate latent space prior to finetune
        if args.finetune_latentspace:
            # get a test dataset with augmented samples
            ft_test_lat_dset = test_loader
            ft_test_lat_dset.dataset.fine_tune_mode = True
            ft_test_lat_loader = DataLoader(ft_test_lat_dset, batch_size = args.target_batch_size)
            # evaluate latent space for test, val and train set
            outputs_val = evaluate_latent_space(model = model, data_loader = finetune_val_loader, device = device, classifier = args.train_classifier, save_h = False)
            outputs_test = evaluate_latent_space(model = model, data_loader = ft_test_lat_loader, device = device, classifier = args.train_classifier, save_h = False)
            outputs_train = evaluate_latent_space(model = model, data_loader = finetune_loader, device = device, classifier = args.train_classifier, save_h = False)
            with open(f'{output_path}/prior_finetune_val_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_val, file)
            with open(f'{output_path}/prior_finetune_test_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_test, file)
            with open(f'{output_path}/prior_finetune_train_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_train, file)

        Classifier = ClassifierModule(num_classes, avg_channels=avg_channels_after)
        Classifier.to(device)
        class_optimizer = Adam(Classifier.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        time = datetime.datetime.now()  
        # define optimizer for encoder, if not frozen
        if args.optimize_encoder:
            optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = None
        # finetune model
        model, losses = finetune_model(model = model, 
                                        classifier = Classifier, 
                                        data_loader = finetune_loader, 
                                        val_loader = finetune_val_loader,
                                        loss_fn = loss_fn, 
                                        optimizer = optimizer, 
                                        class_optimizer = class_optimizer, 
                                        epochs = args.finetune_epochs, 
                                        device = device,
                                        lambda_ = 0.2, 
                                        delta = args.delta)
        time2 = datetime.datetime.now()     
        print('Finetuning the model for', args.finetune_epochs,'epochs took', time2-time, 's.')

        # evaluate latent space post finetune
        if args.finetune_latentspace:
            outputs_val = evaluate_latent_space(model = model, data_loader = finetune_val_loader, device = device, classifier = args.train_classifier, save_h = False)
            outputs_test = evaluate_latent_space(model = model, data_loader = ft_test_lat_loader, device = device, classifier = args.train_classifier, save_h = False)
            outputs_train = evaluate_latent_space(model = model, data_loader = finetune_loader, device = device, classifier = args.train_classifier, save_h = False)
            with open(f'{output_path}/post_finetune_val_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_val, file)
            with open(f'{output_path}/post_finetune_test_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_test, file)
            with open(f'{output_path}/post_finetune_train_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_train, file)
        time = time2
        # evaluate on test set
        test_loader.fine_tune_mode = False
        results = evaluate_model(model = model,
                                classifier = Classifier, 
                                test_loader = test_loader,
                                device = device)
        time2 = datetime.datetime.now()     
        print('Evaluating the finetuned model took', time2-time, 's.')

        # plot and save losses
        plot_contrastive_losses(losses['train'], f'{output_path}/finetune_train_loss_optenc_{args.optimize_encoder}_{finetune_dset}.png')
        plot_contrastive_losses(losses['val'], f'{output_path}/finetune_val_loss_optenc_{args.optimize_encoder}_{finetune_dset}.png')

        # save results
        with open(f'{output_path}/finetune_results_optenc_{args.optimize_encoder}_{finetune_dset}.pickle', 'wb') as file:
            pickle.dump(results, file)
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--train_TFC', type = eval, default = True)
    parser.add_argument('--train_classifier', type = eval, default = True)
    parser.add_argument('--evaluate_latent_space', type = eval, default = False)
    parser.add_argument('--save_model', type = eval, default = True)
    parser.add_argument('--finetune', type = eval, default = True)
    parser.add_argument('--finetune_latentspace', type = eval, default = False)
    parser.add_argument('--optimize_encoder', type = eval, default = True)
    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)

    # data arguments
    parser.add_argument('--test_mode', type = eval, default = True)
    parser.add_argument('--sample_subjs', type = int, default = 12)
    parser.add_argument('--sample_test_subjs', type = int, default = 6)
    parser.add_argument('--config_path', type = str, default = 'sleepeeg_local.yml')
    parser.add_argument('--finetune_path', type = str, default = 'same')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--target_batch_size', type = int, default = 128)
    parser.add_argument('--output_path', type = str, default = 'outputs')

    # augmentation arguments
    parser.add_argument('--abs_budget', type = eval, default = False)
    parser.add_argument('--stride', type = int, default = 4)
    parser.add_argument('--avg_channels', type = str, default = 'None')
    parser.add_argument('--sample_channel', type = eval, default = False)

    # optimizer arguments
    parser.add_argument('--delta', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 3e-6)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--finetune_epochs', type = int, default = 0)
    args = parser.parse_args()
    main(args)

