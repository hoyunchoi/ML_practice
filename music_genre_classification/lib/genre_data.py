import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

"""
    class genre_data

    (public) Methods
    get_train_loader(batch_size, num_workers) : return pytorch data loader for train set
    get_val_loader(batch_size, num_workers) : return pytorch data loader for validation set
    get_test_loader(num_workers) : return pytorch data loader for test set. Batch_size=1

"""

#? Need to add one-hot encoding for various loss functions
class genre_data():
    def __init__(self, data_path, nn_type='FNN', val_size=0.2, test_size=0.25, precision='64', verbose=False):
        # * Get input
        self.precision = precision
        self.nn_type = nn_type

        # * Read data
        self._read_json(data_path)
        if verbose:
            print("Finished reading a file", data_path)
            print('-----------------------------')

        self.input_size = self.input.shape[1] * self.input.shape[2]
        self.output_size = len(self.genres)

        # * Generate Tensor dataset
        self.train_dataset, self.val_dataset, self.test_dataset = self._generate_dataset(val_size, test_size)
        if verbose:
            x,y = self.train_dataset[0]
            print("Number of train set:", len(self.train_dataset), "with input:", x.shape, 'output:', y.shape)
            x,y = self.val_dataset[0]
            print("Number of validation set:", len(self.val_dataset), "with input:", x.shape, 'output:', y.shape)
            x,y = self.test_dataset[0]
            print("Number of test set:", len(self.test_dataset), "with input:", x.shape, 'output:', y.shape)

    def _read_json(self, data_path):
        with open(data_path, "r") as file:
            data = json.load(file)

        # * Convert lists to ndarray and save
        self.genres = np.asanyarray(data["mapping"])
        self.input = np.asanyarray(data["mfcc"], dtype=getattr(np, 'float' + self.precision))
        self.output = np.asanyarray(data["labels"], dtype=np.int64)

    def _generate_dataset(self, val_size, test_size):
        x_train, x_test, y_train, y_test = train_test_split(self.input, self.output, test_size=test_size, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=1)
        x_train, y_train = torch.as_tensor(x_train), torch.as_tensor(y_train)
        x_val, y_val = torch.as_tensor(x_val), torch.as_tensor(y_val)
        x_test, y_test = torch.as_tensor(x_test), torch.as_tensor(y_test)

        #* Add dimension of 'channel' for CNN input
        if self.nn_type == "CNN":
            x_train.unsqueeze_(1)
            x_val.unsqueeze_(1)
            x_test.unsqueeze_(1)

        return TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)

    def get_train_loader(self, batch_size=32, num_workers=4):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_loader

    def get_val_loader(self, batch_size=32, num_workers=4):
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return val_loader

    def get_test_loader(self, num_workers=4):
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        return test_loader