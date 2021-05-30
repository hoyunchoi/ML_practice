import os
import torch
import numpy as np
import pandas as pd
import sklearn.preprocessing
from torch.utils.data import DataLoader

from covid_dataset import covid_dataset

class covid_data():
    def __init__(self,
                 last_day:str,
                 in_features: list,
                 out_features: list = None,
                 precision='32',
                 train_val_boundary: str = '2021-03',
                 past_days: int = 60,
                 successive_days: int = 1,
                 scaler_name='MinMaxScaler',
                 verbose=True):

        #* Store input variables
        self.past_days = past_days
        self.successive_days = successive_days
        self.scaler_name = scaler_name
        self.np_float_dtype = getattr(np, 'float' + precision)
        self.train_val_boundary = pd.Timestamp(train_val_boundary)

        #* Store feature information
        self.in_features = in_features
        self.out_features = in_features if out_features is None else out_features
        self.common_features = set(self.in_features).intersection(self.out_features)
        self.common_input_idx, self.common_output_idx = self._get_common_idx()

        #* Read data from input file name with parsing date
        self.data_frame = self._read_data_frame(last_day)
        self.in_features_idx, self.out_features_idx = self._get_feature_idx()
        if verbose:
            print("Finished reading a file", last_day)
            print('-----------------------------')

        #* Divide training and validation, test from raw data frame
        self.train_raw, self.val_raw = self._divide_raw_data()

        #* Generate input output data for RNN structure
        self.val_output_scaler = self._save_val_output_scaler()
        self.train_raw, self.train_scaler = self._scale_raw(self.train_raw)
        self.val_raw, self.val_scaler = self._scale_raw(self.val_raw)

        #* Generate input output data for RNN structure
        self.train_dataset = covid_dataset(self.train_raw, self.past_days, self.successive_days, self.in_features_idx, self.out_features_idx)
        self.val_dataset = covid_dataset(self.val_raw, self.past_days, self.successive_days, self.in_features_idx, self.out_features_idx)

        #* Print information of data
        if verbose:
            x, y = self.train_dataset[0]
            print("Number of train set:", len(self.train_dataset), "with input:", x.shape, "output:", y.shape)
            x, y = self.val_dataset[0]
            print("Number of validation set:", len(self.val_dataset), "with input:", x.shape, "output:", y.shape)



    #* Return pytorch data loader for train set
    def get_train_loader(self, batch_size=32, num_workers=4):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return train_loader

    #* Return pytorch data loader for validation set
    def get_val_loader(self, batch_size=32, num_workers=4):
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return val_loader

    #* Only read file for in_features and out_features
    def _read_data_frame(self, last_day):
        df = pd.read_parquet(path=os.path.join('data', last_day+'.parquet.snappy'))
        return df

    #* Get index of in/out features at data_frame
    def _get_feature_idx(self):
        in_features_idx = [self.data_frame.columns.get_loc(feature) for feature in self.in_features]
        out_features_idx = [self.data_frame.columns.get_loc(feature) for feature in self.out_features]
        return in_features_idx, out_features_idx

    #* Get index of common_features at in_feautures and out_features
    def _get_common_idx(self):
        common_input_idx, common_output_idx = [], []
        for feature in self.common_features:
            common_input_idx.append(self.in_features.index(feature))
            common_output_idx.append(self.out_features.index(feature))
        return common_input_idx, common_output_idx

    #* Divide raw data into train/validation/test raw data
    def _divide_raw_data(self):
        #* Get start/end (number) index of train/validation/test set
        train_raw_start, train_raw_end = 0, len(self.data_frame[:self.train_val_boundary])
        val_raw_start, val_raw_end = train_raw_end - self.past_days, len(self.data_frame)

        #* Get train/validation/test set
        train_raw = np.asarray(self.data_frame.iloc[:][train_raw_start: train_raw_end].values, dtype=self.np_float_dtype)
        val_raw = np.asarray(self.data_frame.iloc[:][val_raw_start: val_raw_end].values, dtype=self.np_float_dtype)
        return train_raw, val_raw

    #* save scaler of test output and test output tensor
    def _save_val_output_scaler(self):
        #* Scale test output and save the scaler
        scaler = getattr(sklearn.preprocessing, self.scaler_name)()
        scaler.fit_transform(self.val_raw[self.past_days:, self.out_features_idx])
        return scaler

    #* Scale the raw data
    def _scale_raw(self, raw):
        scaler = getattr(sklearn.preprocessing, self.scaler_name)()
        raw = scaler.fit_transform(raw)
        return raw, scaler
