from typing import List, Iterator, Tuple
from ..core import Node, ApplyToDataUnit, DataUnit, Data


# class Trainer(Operator):
#     def __init__(self, model,
#                        epochs,
#                        batch_size,
#                        transforms,
#                        **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.transforms = transforms

#         self.best_valid_error = float('inf')

#     # def _set_model(self):
#         # pass



#     def batch_sampler(self, indxs): 
#         indx_sampler = torch.utils.data.SubsetRandomSampler(indxs)
#         batch_sampler = torch.utils.data.BatchSampler(indx_sampler, self.batch_size, drop_last=False)
#         return batch_sampler


#     def training_epoch(self, train_loader):
#         self.model.train()
        
#         total_loss = 0
#         for x, y in tqdm(train_loader):
#             step_loss = self.training_step(x, y)
#             total_loss += step_loss
            
#         train_error = total_loss / len(train_loader)
#         return train_error

    
#     def validation_epoch(self, valid_loader):
#         self.model.eval()
#         total_valid_error = 0
#         with torch.no_grad():
#             for x, y in tqdm(valid_loader):
#                 valid_error = self.validation_step(x, y)
#                 total_valid_error += valid_error
                
#             valid_error = total_valid_error / len(valid_loader)
            
#         return valid_error

#     def fit(self, x: DataUnit, y: DataUnit) -> DataUnit:
#         path_to_tb = './logs' + '/run_' + strftime("%y_%m_%d_%H_%M_%S", gmtime())   
#         writer = SummaryWriter(path_to_tb)
        
#         x_train, x_valid = x['train'], x['valid']
#         y_train, y_valid = y['train'], y['valid']


#         for e in range(self.epochs):
#             print(f'Training Epoch: {e+1}/{self.epochs}')
#             train_error = self.training_epoch(train_loader)
#             valid_error = self.validation_epoch(valid_loader)
#             writer.add_scalars('Loss', {'train': train_error,
#                                         'valid': valid_error}, e)
#         if valid_error < self.best_valid_error:
#             self.best_valid_error = valid_error
#             self.save(e, train_error)

#         writer.close()
#         # y2 = self.predict_forward(x)
#         return x, y

#     @ApplyToDataUnit(mode='efficient')
#     def predict_forward(self, x : DataUnit) -> DataUnit:
#         self.model.eval()
#         data_loader = self.to_dataloader(x, transform=self.transforms['valid'])
#         with torch.no_grad():
#             for batch in tqdm(data_loader):
#                 batch = batch.to(torch.device("cuda"), dtype=torch.float32)
#                 y_pred = self.model(batch)
            

#         return prediction

#     def to_dataloader(self, x, y=None, transform=None):
#         x = x.X #torch.from_numpy(x.X)
#         if y is not None:
#             y = torch.from_numpy(y.Y)
#             tnsr_data = TensorData(x, y, transform)
#         else:
#             tnsr_data = TensorData(x, transform)

#         data_loader = DataLoader(tnsr_data,
#                                 batch_size=self.batch_size,
#                                 shuffle=True, 
#                                 drop_last=False)
#         return data_loader
    
class Trainer(Node):
    def __init__(self, model,
                    epochs,
                    **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.epochs = epochs

    def fit(self, x: DataUnit, y: DataUnit) -> Tuple[DataUnit, DataUnit]:
        x, y = x.X, y.Y
        for e in range(self.epochs):
            print(f'Training Epoch: {e+1}/{self.epochs}')
            x2, y2 = self.model.fit(x, y)
        return x2, y2

    def predict_forward(self, x : DataUnit) -> DataUnit:
        x2 = self.model.predict_forward(x)
        return x2

    def predict_backward(self, y_frwd: DataUnit) -> DataUnit:
        y = self.model.predict_backward(y_frwd)
        return y
