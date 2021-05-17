import torch
import torch.nn as nn

"""
    class FNN
    __init__(self, input_size, output_size, hidden_size1 = 512, hidden_size2 = 256, hidden_size3 = 64)
    Args:
        input_size: flattened length (n_mfcc * num_mfcc_vectors_per_segment)
        output_size: number of labels
"""

class FNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1 = 512, hidden_size2 = 256, hidden_size3 = 64, dropout = 0.3):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size1)
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2)
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=hidden_size3)
        self.fc4 = nn.Linear(in_features=hidden_size3, out_features=output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        x = self.fc4(x)
        return x