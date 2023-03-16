import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.losses import TS2VecLoss
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from copy import deepcopy
import wandb

class DilatedCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size):
        super().__init__()
        padding = ((kernel_size - 1) * dilation + 1)//2
        self.layer = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding = padding),
            nn.GELU(),
            nn.Conv1d(in_channels=out_channels,out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding = padding)
        )
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.projector is None else self.projector(x)
        out = self.layer(x)
        return res + out
    
def generate_binomial_mask(B, T, p = 0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TS2VecClassifer(nn.Module):
    def __init__(self, n_classes, in_features):
        super().__init__()
        self.classifier = nn.Linear(in_features=in_features, out_features=n_classes)
    def forward(self, latents):
        ts_length = latents.shape[2]
        latents = F.max_pool1d(latents, ts_length).squeeze(-1)
        return self.classifier(latents)

class TS2VecEncoder(nn.Module):
    def __init__(self, input_size, hidden_channels, out_dim, nlayers = 10, kernel_size = 3):
        super().__init__()
        self.linear_projection = nn.Linear(in_features=input_size, out_features=hidden_channels)

        in_channels = [hidden_channels]*(nlayers + 1)
        out_channels = [hidden_channels]*nlayers + [out_dim]
        dilation = [2**i for i in range(nlayers+1)]
        convblocks = [DilatedCNNBlock(in_ch, out_ch, dil, kernel_size) for in_ch, out_ch, dil in zip(in_channels, out_channels, dilation)]
        self.convblocks = nn.Sequential(*convblocks)
        self.repr_dropout = nn.Dropout(p = 0.1)
    
    def forward(self, x, train = False):
        x = x.transpose(1,2)
        proj = self.linear_projection(x)
        if self.training:
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
            proj[~mask] = 0
        proj = torch.permute(proj, (0, 2, 1))
        out = self.repr_dropout(self.convblocks(proj))

        return out

    def take_per_row(self, A, indx, num_elem):
        A = A.transpose(1,2)
        all_indx = indx[:,None] + np.arange(num_elem)
        return A[torch.arange(all_indx.shape[0])[:,None],all_indx].transpose(1,2)

    def evaluate_latent_space(self,
                              dataloader):
        self.training = False


    def fit(self, 
            dataloader,
            val_dataloader,
            epochs,
            optimizer,
            alpha,
            temporal_unit,
            device,
            backup_path = None,
            log = True):
        
        loss_fn = TS2VecLoss(alpha = alpha, temporal_unit=temporal_unit)
        loss_collect = []
        temp_loss_collect = []
        inst_loss_collect = []
        val_loss_collect = []
        val_temp_loss_collect = []
        val_inst_loss_collect = []
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_temp_loss = 0
            epoch_inst_loss = 0
            self.training = True
            for i, data in enumerate(dataloader):
                x = data[0].float().to(device)

                # create cropped views
                ts_l = x.size(2)
                crop_l = np.random.randint(low=2, high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))


                optimizer.zero_grad()
                
                out1 = self.forward(self.take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft), train = True)
                out1 = out1[:, :, -crop_l:]
                
                out2 = self.forward(self.take_per_row(x, crop_offset + crop_left, crop_eright - crop_left), train = True)
                out2 = out2[:, :, :crop_l]

                loss, inst_loss, temp_loss = loss_fn(out1, out2)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().cpu().numpy()
                epoch_temp_loss += temp_loss.numpy()
                epoch_inst_loss += inst_loss.numpy()

            loss_collect.append(epoch_loss/(i+1))
            temp_loss_collect.append(epoch_temp_loss/(i+1))
            inst_loss_collect.append(epoch_inst_loss/(i+1))

            epoch_loss = 0
            epoch_temp_loss = 0
            epoch_inst_loss = 0
            self.training = False
            for i, data in enumerate(val_dataloader):
                x = data[0].float().to(device)

                # create cropped views
                ts_l = x.size(2)
                crop_l = np.random.randint(low=2, high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                out1 = self.forward(self.take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft), train = True)
                out1 = out1[:, :, -crop_l:]
                
                out2 = self.forward(self.take_per_row(x, crop_offset + crop_left, crop_eright - crop_left), train = True)
                out2 = out2[:, :, :crop_l]

                loss, inst_loss, temp_loss = loss_fn(out1, out2)

                epoch_loss += loss.detach().cpu().numpy()
                epoch_temp_loss += temp_loss.numpy()
                epoch_inst_loss += inst_loss.numpy()

            val_loss_collect.append(epoch_loss/(i+1))
            val_temp_loss_collect.append(epoch_temp_loss/(i+1))
            val_inst_loss_collect.append(epoch_inst_loss/(i+1))
            if log:
                wandb.log({'pretrain val time loss': val_temp_loss_collect[-1], 
                           'pretrain val inst loss': val_inst_loss_collect[-1], 
                           'pretrain val total loss': val_loss_collect[-1],
                           'pretrain time loss': temp_loss_collect[-1], 
                           'pretrain inst loss': inst_loss_collect[-1], 
                           'pretrain total loss': loss_collect[-1]})
            
            if backup_path is not None and epoch % 2 == 0:
                path = f'{backup_path}/pretrained_model.pt'
                torch.save(self.state_dict(), path)

        losses = {
            'train': 
            {
            'Time loss': temp_loss_collect,
            'Instance loss': inst_loss_collect,
            'Total loss': loss_collect
        },
            'val':
            {
            'Time loss': val_temp_loss_collect,
            'Instance loss': val_inst_loss_collect,
            'Total loss': val_loss_collect
        }
        }
        return losses
    
    def finetune(self, 
                 train_loader,
                 val_loader, 
                 classifier,
                 optimizer,
                 epochs,
                 device,
                 log = True,
                 choose_best = True
                 ):
        self.training = True
        classifier.training = True
        class_loss_fn = torch.nn.CrossEntropyLoss()

        loss_collect = []
        val_loss_collect = []
        if choose_best:
            accuracy = 0
            best_state_dict = deepcopy(self.state_dict())
            best_class_state_dict = deepcopy(classifier.state_dict())

        for epoch in range(epochs):
            epoch_loss = 0
            val_epoch_loss = 0
            self.training = True
            classifier.training = True
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                x, y = batch[0].float().to(device), batch[-1].long().to(device)

                out = self.forward(x)
                pred = classifier(out)
                class_loss = class_loss_fn(pred, y)
                class_loss.backward()
                optimizer.step()
                epoch_loss += class_loss.detach().cpu().numpy()

            loss_collect.append(epoch_loss/(i+1))
            self.training = False
            collect_y = []
            collect_pred = []
            classifier.eval()

            for i, batch in enumerate(val_loader):
                x, y = batch[0].float().to(device), batch[-1].long().to(device)

                out = self.forward(x)
                pred = classifier(out)

                class_loss = class_loss_fn(pred, y)
                val_epoch_loss += class_loss.detach().cpu().numpy()
                collect_y.append(y.detach().cpu().numpy())
                collect_pred.append(torch.argmax(pred, dim = -1).detach().cpu().numpy())

            collect_y = np.hstack(collect_y)
            collect_pred = np.hstack(collect_pred)
            val_accuracy = balanced_accuracy_score(collect_y, collect_pred)
            prec, rec, f, _ = precision_recall_fscore_support(collect_y, collect_pred)
            val_loss_collect.append(val_epoch_loss/(i+1))
            
            if log:
                wandb.log({
                    'train class. loss': loss_collect[-1],
                    'val class. loss': val_loss_collect[-1],
                    'val accuracy': val_accuracy,
                    'val precision': prec,
                    'val recall': rec,
                    'val f1 score': f
                })

            if choose_best:
                if val_accuracy > accuracy:
                    accuracy = val_accuracy
                    best_state_dict = deepcopy(self.state_dict())
                    best_class_state_dict = deepcopy(classifier.state_dict())

        if choose_best:
            self.load_state_dict(best_state_dict)
            classifier.load_state_dict(best_class_state_dict)

        return classifier
    
    def evaluate_classifier(self, data_loader, classifier, device):
        collect_y, collect_pred = [], []
        self.training = False
        for batch in data_loader:
            x, y = batch[0].float().to(device), batch[-1].long()
            out = self.forward(x)
            pred = classifier(out)

            collect_y.append(y.numpy())
            collect_pred.append(pred.detach().cpu().numpy())

        collect_y = np.hstack(collect_y)
        collect_pred = np.hstack(collect_pred)
        accuracy = balanced_accuracy_score(collect_y, collect_pred)
        prec, rec, f, _ = precision_recall_fscore_support(collect_y, collect_pred)

        return accuracy, prec, rec, f



    def evaluate_latent_space(self, data_loader, device, maxpool = True):
        latent_space = []
        collect_y = []
        for i, data in enumerate(data_loader):
            x = data[0].float().to(device)
            output = self.forward(x)
            if maxpool:
                ts_length = x.shape[2]
                output = F.max_pool1d(output, ts_length)
            latent_space.append(output.detach().cpu().numpy())
            collect_y.append(data[-1].unsqueeze(1).numpy())
        
        outputs = {
            'latents': np.vstack(latent_space),
            'y': np.vstack(collect_y)
        }
        return outputs



            





