import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models import wave2vecblock
import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from utils.losses import ContrastiveLoss, TS2VecLoss
import wandb
import copy

class TimeClassifier(nn.Module):
    def __init__(self, in_features, num_classes, pool, orig_channels = 9, time_length = 33):
        super().__init__()
        self.pool = pool
        self.flatten = nn.Flatten()
        self.adpat_avg = nn.AdaptiveAvgPool1d(4)
        
        self.channelreduction = nn.Linear(in_features=orig_channels, out_features=1)
        if self.pool == 'adapt_avg':
            in_features = 4*in_features
        elif self.pool == 'flatten':
            in_features = in_features*time_length
            
        self.classifier = nn.Linear(in_features=in_features, out_features=num_classes)
    def forward(self, latents):
        if len(latents.shape) > 3:
            latents = latents.permute(0,2,3,1)
            latents = self.channelreduction(latents).squeeze(-1)

        ts_length = latents.shape[2]
        if self.pool == 'max':
            latents = F.max_pool1d(latents, ts_length).squeeze(-1)
        elif self.pool == 'last':
            latents = latents[:,:,-1]
        elif self.pool == 'avg':
            latents = F.avg_pool1d(latents, ts_length).squeeze(-1)
        elif self.pool == 'adapt_avg':
            latents = self.flatten(self.adpat_avg(latents))
        else:
            latents = self.flatten(latents)

        return self.classifier(latents)

class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Linear(in_features=in_features, out_features=num_classes)
    def forward(self, x):
        return self.classifier(x)


def conv1D_out_shape(input_shape, kernel, stride, padding):
    kernel, stride, padding = np.array(kernel), np.array(stride), np.array(padding)
    shape = input_shape
    for kern, stri, padd in zip(kernel, stride, padding):
        shape = int((shape + 2*padd - kern)/stri + 1)
    return shape

class Wave2Vec(nn.Module):
    def __init__(self, channels, input_shape, out_dim = 64, hidden_channels = 512, nlayers = 6, norm = 'group'):
        super().__init__()
        self.channels = channels
        width = [3] + [2]*(nlayers-1)
        in_channels = [channels] + [hidden_channels]*(nlayers-1)
        out_channels = [hidden_channels]*(nlayers - 1) + [out_dim]
        self.convblocks = nn.Sequential(*[wave2vecblock(channels_in= in_channels[i], channels_out = out_channels[i], kernel = width[i], stride = width[i], norm = norm) for i in range(nlayers)])
        self.out_shape = conv1D_out_shape(input_shape, width, width, [w//2 for w in width])
    def forward(self, x):
        return self.convblocks(x)



class GNNMultiview(nn.Module):
    def __init__(self, 
                 channels, 
                 time_length, 
                 num_classes, 
                 flatten = True,
                 norm = 'group', 
                 num_message_passing_rounds = 3, 
                 hidden_channels = 512, 
                 nlayers = 6, 
                 out_dim = 64,
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.time_length = time_length
        self.num_classes = num_classes
        self.flatten = flatten
        self.wave2vec = Wave2Vec(channels, input_shape = time_length, out_dim = out_dim, hidden_channels = hidden_channels, nlayers = nlayers, norm = norm)

        if self.flatten:
            out_dim = self.wave2vec.out_shape * out_dim
            self.classifier = Classifier(in_features=out_dim, num_classes=num_classes)
        else:
            out_dim = out_dim
            self.classifier = TimeClassifier(in_features = out_dim, num_classes = num_classes, pool = 'adapt_avg', orig_channels = channels, time_length = time_length)

        self.state_dim = out_dim
        self.flat = nn.Flatten()
        print('out_dim', out_dim)
        

        # Message passing layers (from_state, to_state) -> message
        self.message_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim*2, out_dim),
                nn.Tanh()
            )
            for _ in range(num_message_passing_rounds)
        ])

        # Readout layer
        self.readout_net = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Tanh()
        )

    def forward(self, x, classify = False):
        b, ch, ts = x.shape
        x = x.view(b*ch, 1, ts)
        latents = self.wave2vec(x)

        view_id = torch.arange(b).unsqueeze(1).repeat(1, ch).view(-1).to(x.device)
        message_from = torch.arange(b*ch).unsqueeze(1).repeat(1, (ch-1)).view(-1).to(x.device)
        message_to = torch.arange(b*ch).view(b, ch).unsqueeze(1).repeat(1, ch, 1)
        idx = ~torch.eye(ch).view(1, ch, ch).repeat(b, 1, 1).bool()
        message_to = message_to[idx].view(-1).to(x.device)

        if self.flatten:
            latents = self.flat(latents)
        else:
            latents = latents.permute(0,2,1)

        for message_net in self.message_nets:
            message = message_net(torch.cat([latents[message_from], latents[message_to]], dim=-1))
            # Sum messages
            latents.index_add_(0, message_to.to(x.device), message)

        y = torch.zeros(b, *latents.shape[1:]).to(x.device)
        y.index_add_(0, view_id, latents)
        out = self.readout_net(y)

        if not self.flatten:
            out = out.permute(0,2,1)

        if classify:
            return self.classifier(out)
        else:
            return out
        
    def get_view_pairs(self, view_id):
        """Yield all distinct pairs."""
        for i, vi in enumerate(view_id):
            for j, vj in enumerate(view_id):
                if vi == vj and i != j:
                    yield [i, j]

    def take_channel(self, A, indx):
        indx = indx.unsqueeze(2).repeat(1,1,A.shape[2])
        return torch.gather(A, 1, indx)

    def train_step(self, x, loss_fn, device):
        # partition the dataset into two views
        ch_size = np.random.randint(2, x.size(1)-1)
        random_channels = np.random.rand(x.size(0), x.size(1)).argpartition(ch_size,axis=1)
        view_1_idx = random_channels[:,:ch_size] # randomly select ch_size channels per input
        view_2_idx = random_channels[:,ch_size:] # take the remaining as the second view
        view_1 = self.take_channel(x, torch.tensor(view_1_idx).to(device))
        view_2 = self.take_channel(x, torch.tensor(view_2_idx).to(device))
        out1 = self.forward(view_1)
        out2 = self.forward(view_2)
        loss = loss_fn(out1, out2)
        if isinstance(loss, tuple):
            return loss
        else:
            return loss, *[0]*(len(loss)-1)


    def fit(self, 
            dataloader,
            val_dataloader,
            epochs,
            optimizer,
            device,
            time_loss = False,
            temperature = 0.5,
            backup_path = None,
            log = True):
        
        if time_loss:
            loss_fn = TS2VecLoss(alpha = 0.5, temporal_unit = 0).to(device)
        else:
            loss_fn = ContrastiveLoss(device, temperature).to(device)
        self.to(device)
        for epoch in range(epochs):
            epoch_loss = 0
            self.train()
            for i, data in enumerate(dataloader):
                x = data[0].to(device).float()
                optimizer.zero_grad()
                loss, inst_loss, temp_loss = self.train_step(x, loss_fn, device)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss/(i+1)
            val_loss = 0
            self.eval()
            for i, data in enumerate(val_dataloader):
                x = data[0].to(device).float()
                loss = self.train_step(x, loss_fn, device)
                val_loss += loss.item()
            if log:
                log_dict = {'val_loss': val_loss/(i+1), 'train_loss': train_loss}
                if time_loss:
                    log_dict['inst_loss'] = inst_loss
                    log_dict['temp_loss'] = temp_loss
                wandb.log(log_dict)

            if backup_path is not None:
                path = f'{backup_path}/pretrained_model.pt'
                torch.save(self.state_dict(), path)
    
    def finetune(self,
                 dataloader,
                 val_dataloader,
                 epochs,
                 optimizer,
                 weights,
                 device,
                 choose_best = True,
                 backup_path = None):
        self.to(device)
        loss = nn.CrossEntropyLoss(weight=weights)
        best_accuracy = 0
        best_model = self.state_dict()
        for epoch in range(epochs):
            epoch_loss = 0
            self.train()
            for i, data in enumerate(dataloader):
                x = data[0].to(device).float()
                y = data[-1].to(device)
                optimizer.zero_grad()
                out = self.forward(x, classify = True)
                loss_ = loss(out, y)
                loss_.backward()
                optimizer.step()
                epoch_loss += loss_.item()
            train_loss = epoch_loss/(i+1)
            val_loss = 0
            collect_y = []
            collect_pred = []

            self.eval()
            for i, data in enumerate(val_dataloader):
                x = data[0].to(device).float()
                y = data[-1].to(device)
                out = self.forward(x, classify = True)
                loss_ = loss(out, y)
                val_loss += loss_.item()
                collect_y.append(y.detach().cpu().numpy())
                collect_pred.append(out.argmax(dim=1).detach().cpu().numpy())
            collect_y = np.concatenate(collect_y)
            collect_pred = np.concatenate(collect_pred)
            acc = balanced_accuracy_score(collect_y, collect_pred)
            prec, rec, f, _ = precision_recall_fscore_support(collect_y, collect_pred)

            wandb.log({'train_class_loss': train_loss, 'val_class_loss': val_loss/(i+1), 'val_acc': acc, 'val_prec': prec, 'val_rec': rec, 'val_f': f})
            if choose_best:
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model = copy.deepcopy(self.state_dict())

            if backup_path is not None:
                path = f'{backup_path}/finetuned_model.pt'
                torch.save(self.state_dict(), path)

        if choose_best:
            self.load_state_dict(best_model)

    def evaluate_classifier(self,
                            test_loader,
                            device):
        self.eval()
        collect_y = []
        collect_pred = []
        for i, data in enumerate(test_loader):
            x = data[0].to(device).float()
            y = data[-1].to(device)
            out = self.forward(x, classify = True)
            collect_y.append(y.detach().cpu().numpy())
            collect_pred.append(out.argmax(dim=1).detach().cpu().numpy())
        collect_y = np.concatenate(collect_y)
        collect_pred = np.concatenate(collect_pred)
        acc = balanced_accuracy_score(collect_y, collect_pred)
        prec, rec, f, _ = precision_recall_fscore_support(collect_y, collect_pred)
        return acc, prec, rec, f