import torch
from torch import Tensor

import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from time import gmtime, strftime
from ..core import Operator, ApplyToDataUnit, DataUnit, Data


class Trainer(Operator):
    def __init__(self, model,
                       optimizer, 
                       loss_func, 
                       epochs,
                       batch_size,
                       transforms,
                       **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.transforms = transforms

        self.best_valid_error = float('inf')

    # def _set_model(self):
        # pass

    def transform(self, x, y):
        x_new = x.to(torch.device("cuda"), dtype=torch.float32)
        y_new = y.to(torch.device("cuda"), dtype=torch.long)
        return x_new, y_new

    # def batch_sampler(self, indxs): 
    #     indx_sampler = torch.utils.data.SubsetRandomSampler(indxs, generator=None)
    #     batch_sampler = torch.utils.data.BatchSampler(indx_sampler, self.batch_size, drop_last=True)
    #     return batch_sampler


    def training_step(self, x, y):
        self.optimizer.zero_grad()
        x, y = self.transform(x, y)
        y_pred = self.model(x)
        loss =  self.loss_func(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def training_epoch(self, train_loader):
        self.model.train()
        
        total_loss = 0
        for x, y in tqdm(train_loader):
            step_loss = self.training_step(x, y)
            total_loss += step_loss
            
        train_error = total_loss / len(train_loader)
        return train_error

    def validation_step(self, x, y):
        x, y = self.transform(x, y)
        y_pred = self.model(x)

        # prob = F.softmax(y_pred, dim=1)
        # label = torch.argmax(prob, dim=1)
        # acc = (label == y).sum()
        return F.cross_entropy(y_pred, y)
    
    def validation_epoch(self, valid_loader):
        self.model.eval()
        total_valid_error = 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader):
                valid_error = self.validation_step(x, y)
                total_valid_error += valid_error
                
            valid_error = total_valid_error / len(valid_loader)
            
        return valid_error

    def fit(self, x: DataUnit, y: DataUnit) -> DataUnit:
        path_to_tb = './logs' + '/run_' + strftime("%y_%m_%d_%H_%M_%S", gmtime())   
        writer = SummaryWriter(path_to_tb)
        
        x_train, x_valid = x['train'], x['valid']
        y_train, y_valid = y['train'], y['valid']

        train_loader = self.to_dataloader(x_train, y_train, self.transforms['train'])
        valid_loader = self.to_dataloader(x_valid, y_valid, self.transforms['valid'])

        for e in range(self.epochs):
            print(f'Training Epoch: {e+1}/{self.epochs}')
            train_error = self.training_epoch(train_loader)
            valid_error = self.validation_epoch(valid_loader)
            writer.add_scalars('Loss', {'train': train_error,
                                        'valid': valid_error}, e)
        if valid_error < self.best_valid_error:
            self.best_valid_error = valid_error
            self.save(e, train_error)

        writer.close()
        # y2 = self.predict_forward(x)
        return x, y

    @ApplyToDataUnit(mode='efficient')
    def predict_forward(self, x : DataUnit) -> DataUnit:
        self.model.eval()
        data_loader = self.to_dataloader(x, transform=self.transforms['valid'])
        with torch.no_grad():
            for batch in tqdm(data_loader):
                batch = batch.to(torch.device("cuda"), dtype=torch.float32)
                y_pred = self.model(batch)
            

        return prediction

    def to_dataloader(self, x, y=None, transform=None):
        x = x.X #torch.from_numpy(x.X)
        if y is not None:
            y = torch.from_numpy(y.Y)
            tnsr_data = TensorData(x, y, transform)
        else:
            tnsr_data = TensorData(x, transform)

        data_loader = DataLoader(tnsr_data,
                                batch_size=self.batch_size,
                                shuffle=True, 
                                drop_last=False)
        return data_loader
    
    def save(self, epoch, loss):
        prefix = 'models/'
        file_name='model_weights'
        suffix = strftime("%y_%m_%d_%H_%M_%S", gmtime())
        ext = '.pth'
        path = prefix + file_name + '_' + suffix + ext
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    },
                    path)


# class TensorData(Dataset):
#     def __init__(self, data: Tensor, target: Tensor, transform=None):
#         assert data.shape[0] == target.size(0), 'Size mismatch between tensors.'
#         self.data = data
#         self.target = target
#         self.transform = transform
        
#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.target[index]

#         if self.transform:
#             x = self.transform(image=x)['image']
        
#         return x, y
    
#     def __len__(self):
#         return len(self.data)


class TensorData(Dataset):
    def __init__(self, *tensors: Tensor, transform=None) -> None:
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(image=x)['image']
        return x, tuple(tensor[index] for tensor in self.tensors[1:])

    def __len__(self):
        return self.tensors[0].size(0)