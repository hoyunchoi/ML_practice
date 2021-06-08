import os
import typing
import torch
import numpy as np
import pandas as pd
import argparse
import sklearn.preprocessing
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset

from stock_dataset import stock_dataset
"""
    class stock_data

    (public) Members
    in_features : Feature names of input for RNN
    out_features : Feature names of output from RNN
    train_raw : raw stock data for train set. Scaled by train_scaler if scaler_name is given
    val_raw : raw stock data for validation set. Scaled by val_scaler if scaler_name is given
    test_raw : raw stock data for test set. Scaled by test_scaler if scaler_name is given
    test_input : input data for RNN, torch.tensor(sample_num, past_days, in_features)
    test_output : real output data for RNN, torch.tensor(sample_num, 1, in_features)

    (public) Methods
    get_train_loader(batch_size, num_workers) : return pytorch data loader for train set
    get_val_loader(batch_size, num_workers) : return pytorch data loader for validation set
    get_test_loader(num_workers) : return pytorch data loader for test set. Batch_size=1
"""


class stock_data():
    def __init__(self, param: argparse.Namespace,
                 train_val_test_boundary: typing.List[str] = ['2014', '2016'],
                 fmt: str = 'parquet',
                 comp: str = 'snappy',
                 pre_process: bool = True,
                 use_dataset: bool = True,
                 verbose: bool = False):
        #* Store input variables
        self.past_days = param.past_days
        self.successive_days = param.successive_days
        self.scaler_name = param.scaler_name
        self.precision = param.precision
        self.np_float_dtype = getattr(np, 'float' + self.precision)

        #* Get file path of data
        if pre_process:
            self.file = os.path.join("data", "pre_" + self.precision, '.'.join([param.stocks, fmt, comp]))
        else:
            self.file = os.path.join("data", self.stocks + "_2006-01-01_to_2018-01-01.csv")

        #* Boundary of train/validation and validation/test
        self.train_val_boundary = pd.Timestamp(train_val_test_boundary[0])
        self.val_test_boundary = pd.Timestamp(train_val_test_boundary[1])

        #* Store feature information
        self.in_features = param.in_features
        self.out_features = param.out_features
        self.common_features = set(self.in_features).intersection(self.out_features)
        self.common_input_idx, self.common_output_idx = self._get_common_idx()

        #* Read data from input file name with parsing date
        self.data_frame = self._read_data_frame(pre_process, fmt, verbose)
        self.in_features_idx, self.out_features_idx = self._get_feature_idx()
        if verbose:
            print("Finished reading a file", self.file)
            print('-----------------------------')

        #* Divide training and validation, test from raw data frame
        self.train_raw, self.val_raw, self.test_raw = self._divide_raw_data()

        #* Scale the data if scaler is given: only train and validation data
        self.test_output_scaler = self._save_test_output_scaler()
        if self.scaler_name:
            self.train_raw = self._scale_raw_data(self.train_raw, 'train')
            self.val_raw = self._scale_raw_data(self.val_raw, 'validation')
            self.test_raw = self._scale_raw_data(self.test_raw, 'test')

        #* Generate input output data for RNN structure
        if use_dataset:
            self.train_dataset = stock_dataset(param, self.train_raw, self.in_features_idx, self.out_features_idx)
            self.val_dataset = stock_dataset(param, self.val_raw, self.in_features_idx, self.out_features_idx)
            self.test_dataset = stock_dataset(param, self.test_raw, self.in_features_idx, self.out_features_idx)
        elif np.__version__ >= '1.20.0':
            self.train_dataset = self._generate_rnn_dataset_from_raw_stride(self.train_raw)
            self.val_dataset = self._generate_rnn_dataset_from_raw_stride(self.val_raw)
            self.test_dataset = self._generate_rnn_dataset_from_raw_stride(self.test_raw)
        else:
            print("Updating numpy to version 1.20.0 or higher improves performance")
            self.train_dataset = self._generate_rnn_dataset_from_raw(self.train_raw)
            self.val_dataset = self._generate_rnn_dataset_from_raw(self.val_raw)
            self.test_dataset = self._generate_rnn_dataset_from_raw(self.test_raw)

        #* Print information of data
        if verbose:
            x, y = self.train_dataset[0]
            print("Number of train set:", len(self.train_dataset), "with input:", x.shape, "output:", y.shape)
            x, y = self.val_dataset[0]
            print("Number of validation set:", len(self.val_dataset), "with input:", x.shape, "output:", y.shape)
            x, y = self.test_dataset[0]
            print("Number of test set:", len(self.test_dataset), "with input:", x.shape, "output:", y.shape)

    #* Return pytorch data loader for train set
    def get_train_loader(self, batch_size: int = 64,
                         num_workers: int = 4) -> DataLoader:
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
        return train_loader

    #* Return pytorch data loader for validation set
    def get_val_loader(self, batch_size: int = 64,
                       num_workers: int = 4) -> DataLoader:
        val_loader = DataLoader(self.val_dataset,
                                batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
        return val_loader

    #* Return pytorch data loader for test set
    def get_test_loader(self) -> DataLoader:
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True)
        return test_loader

    #* Only read file for in_features and out_features
    def _read_data_frame(self, pre_process: bool, fmt: str, verbose: bool) -> pd.DataFrame:
        read_column = list(set(['Date'] + self.in_features + self.out_features))
        if pre_process:
            if fmt == 'parquet':
                df = pd.read_parquet(self.file, columns=read_column)
            else:
                raise Exception("Given fmt for pre-processed data is invalid")
        else:
            df = pd.read_csv(self.file, usecols=read_column, index_col='Date', parse_dates=['Date'], dtype=self.np_float_dtype, verbose=verbose)
        return df

    #* Get index of in/out features at data_frame
    def _get_feature_idx(self) -> typing.Tuple[list]:
        in_features_idx = [self.data_frame.columns.get_loc(feature) for feature in self.in_features]
        out_features_idx = [self.data_frame.columns.get_loc(feature) for feature in self.out_features]
        return in_features_idx, out_features_idx

    #* Get index of common_features at in_feautures and out_features
    def _get_common_idx(self) -> typing.Tuple[list]:
        common_input_idx, common_output_idx = [], []
        for feature in self.common_features:
            common_input_idx.append(self.in_features.index(feature))
            common_output_idx.append(self.out_features.index(feature))
        return common_input_idx, common_output_idx

    #* Divide raw data into train/validation/test raw data
    def _divide_raw_data(self) -> typing.Tuple[np.ndarray]:
        #* Get start/end (number) index of train/validation/test set
        train_raw_start, train_raw_end = 0, len(self.data_frame[:self.train_val_boundary])
        val_raw_start, val_raw_end = train_raw_end - self.past_days, len(self.data_frame[:self.val_test_boundary])
        test_raw_start, test_raw_end = val_raw_end - self.past_days, len(self.data_frame)

        #* Get train/validation/test set
        train_raw = self.data_frame.iloc[:][train_raw_start: train_raw_end].values
        val_raw = self.data_frame.iloc[:][val_raw_start: val_raw_end].values
        test_raw = self.data_frame.iloc[:][test_raw_start: test_raw_end].values
        return train_raw, val_raw, test_raw

    #* Return scaled data
    def _scale_raw_data(self, raw: np.ndarray, name: str) -> np.ndarray:
        scaler = getattr(sklearn.preprocessing, self.scaler_name)()
        raw = scaler.fit_transform(raw)
        if name == "train":
            self.train_scaler = scaler
        elif name == "validation":
            self.val_scaler = scaler
        else:
            self.test_scaler = scaler
        return raw

    #* save scaler of test output and test output tensor
    def _save_test_output_scaler(self):
        #* Scale test output and save the scaler
        scaler = getattr(sklearn.preprocessing, self.scaler_name)()
        scaler.fit_transform(self.test_raw[self.past_days:, self.out_features_idx])
        return scaler

    def _generate_rnn_dataset_from_raw(self, raw: np.ndarray) -> Dataset:
        #* Divide into input(x) and output(y)
        x = np.zeros((len(raw) - self.past_days - self.successive_days + 1, self.past_days, len(self.in_features)), dtype=self.np_float_dtype)
        y = np.zeros((len(raw) - self.past_days - self.successive_days + 1, self.successive_days, len(self.out_features)), dtype=self.np_float_dtype)

        for i in range(len(raw) - self.past_days - self.successive_days + 1):
            x[i] = raw[i: i + self.past_days, self.in_features_idx]
            y[i] = raw[i + self.past_days: i + self.past_days + self.successive_days, self.out_features_idx]
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        return TensorDataset(x, y)

    def _generate_rnn_dataset_from_raw_stride(self, raw: np.ndarray) -> Dataset:
        #* If input is test set, only calculate input since we have already done at _save_test_output_scaler
        x = np.lib.stride_tricks.sliding_window_view(raw, self.past_days, axis=0)
        y = np.lib.stride_tricks.sliding_window_view(raw, self.successive_days, axis=0)
        x = np.transpose(x[:-1, self.in_features_idx], axes=[0, 2, 1])
        y = np.transpose(y[:, self.out_features_idx], axes=[0, 2, 1])
        if self.successive_days > 1:
            x = torch.as_tensor(x[:-self.successive_days + 1])
        else:
            x = torch.as_tensor(x)
        y = torch.as_tensor(y[self.past_days:])
        return TensorDataset(x, y)


if __name__ == "__main__":
    print("This is module stock_data")
