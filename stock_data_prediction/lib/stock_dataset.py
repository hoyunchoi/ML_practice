import torch
from torch.utils.data import Dataset, DataLoader


class stock_dataset(Dataset):
    def __init__(self, scaled_raw, past_days, successive_days, in_features_idx, out_features_idx):
        self.scaled_raw = torch.as_tensor(scaled_raw)
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


if __name__ == "__main__":
    print("This is module stock_dataset")