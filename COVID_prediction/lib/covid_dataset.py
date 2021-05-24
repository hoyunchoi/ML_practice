import numpy as np
from torch.utils.data import Dataset

class covid_dataset(Dataset):
    def __init__(self,
                 scaled_raw: np.ndarray,
                 past_days: int,
                 successive_days: int,
                 in_features_idx: list,
                 out_features_idx: list):

        #* Store input variables
        self.scaled_raw = scaled_raw
        self.past_days = past_days
        self.successive_days = successive_days
        self.in_features_idx = in_features_idx
        self.out_features_idx = out_features_idx

    def __len__(self):
        return len(self.scaled_raw) - self.past_days - self.successive_days + 1

    def __getitem__(self, idx):
        x = self.scaled_raw[idx : idx+self.past_days, self.in_features_idx]
        y = self.scaled_raw[idx + self.past_days : idx + self.past_days + self.successive_days, self.out_features_idx]
        return x, y