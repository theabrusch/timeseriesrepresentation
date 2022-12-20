import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from utils.trainer import TFC_trainer
from utils.models import TFC_encoder, ContrastiveLoss
from utils.dataset import TFC_Dataset
from torch.optim import Adam
import pickle


def main(args):

    train = torch.load(args.data_path + 'train.pt')
    val = torch.load(args.data_path + 'val.pt')
    test = torch.load(args.data_path + 'test.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TFC_dset = TFC_Dataset(train['samples'], train['labels'])
    train_loader = DataLoader(TFC_dset, batch_size = args.batch_size, shuffle = True, drop_last=True)

    val_dset = TFC_Dataset(val['samples'], val['labels'])
    test_dset = TensorDataset(test['samples'], test['labels'])
    val_loader = DataLoader(val_dset, batch_size = args.batch_size)
    test_loader = DataLoader(test_dset, batch_size = args.batch_size)

    model = TFC_encoder(in_channels = TFC_dset.channels, input_size = TFC_dset.time_length, 
                        num_classes = TFC_dset.num_classes, classify = args.train_classifier)
    model.to(device)

    optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

    loss_fn = ContrastiveLoss(tau = 0.2, batchsize = args.batch_size, device = device)

    model, losses = TFC_trainer(model = model, 
                                train_loader = train_loader, 
                                optimizer = optimizer, 
                                loss_fn = loss_fn, 
                                epochs = args.epochs, 
                                val_loader = val_loader, 
                                device = device, 
                                train_classifier = args.train_classifier)
    
    with open('outputs.pickle', 'wb') as file:
        pickle.dump(losses, file)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = '/work3/theb/timeseries/ECG/')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--train_classifier', type = bool, default = True)
    parser.add_argument('--learning_rate', type = float, default = 3e-4)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)

    parser.add_argument('--epochs', type = int, default = 100)
    args = parser.parse_args()
    main(args)

