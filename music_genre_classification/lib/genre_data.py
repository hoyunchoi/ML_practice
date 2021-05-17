import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

"""
    class genre_data

    (public) Members

"""


class genre_data():
    def __init__(self, data_path, val_size=0.3, precision='64', verbose=False):
        # * Get dtype
        self.precision = precision

        # * Read data
        self._read_json(data_path)
        if verbose:
            print("Finished reading a file", data_path)
            print('-----------------------------')

        self.input_size = self.input.shape[1] * self.input.shape[2]
        self.output_size = len(self.genres)

        # * Generate Tensor dataset
        self.train_dataset, self.val_dataset = self._generate_dataset(val_size)
        if verbose:
            x,y = self.train_dataset[0]
            print("Number of train set:", len(self.train_dataset), "with input:", x.shape, 'output:', y.shape)
            x,y = self.val_dataset[0]
            print("Number of validation set:", len(self.val_dataset), "with input:", x.shape, 'output:', y.shape)

    def _read_json(self, data_path):
        with open(data_path, "r") as file:
            data = json.load(file)

        # * Convert lists to ndarray and save
        self.genres = np.asanyarray(data["mapping"])
        self.input = np.asanyarray(data["mfcc"], dtype=getattr(np, 'float' + self.precision))
        self.output = np.asanyarray(data["labels"], dtype=getattr(np, 'int' + self.precision))

    def _generate_dataset(self, val_size):
        x_train, x_val, y_train, y_val = train_test_split(self.input, self.output, test_size=val_size, random_state=1)
        x_train, x_val, y_train, y_val = torch.as_tensor(x_train), torch.as_tensor(x_val), torch.as_tensor(y_train), torch.as_tensor(y_val)

        return TensorDataset(x_train, y_train), TensorDataset(x_val, y_val)

    def get_train_loader(self, batch_size=32, num_workers=4):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_loader

    def get_val_loader(self, batch_size=2, num_workers=4):
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return val_loader
