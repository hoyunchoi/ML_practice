import torch
import torch.nn as nn
import numpy as np

"""
    class FNN
    __init__(self, input_size, output_size, hidden_size1 = 512, hidden_size2 = 256, hidden_size3 = 64)
    Args:
        input_size: flattened length (n_mfcc * num_mfcc_vectors_per_segment)
        output_size: number of labels
"""


#? Use module list for CNN

class FNN(nn.Module):
    def __init__(self, input_shape, output_size,
                 hidden_size1=512, hidden_size2=256, hidden_size3=64,
                 dropout=0.3):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_shape[1] * input_shape[2], out_features=hidden_size1)
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


class CNN(nn.Module):
    def __init__(self, input_shape, output_size,
                 kernel_size1=(3, 3), stride1=(2, 2),
                 kernel_size2=(3, 3), stride2=(2, 2),
                 kernel_size3=(2, 2), stride3=(2, 2),
                 hidden_channels=32, hidden_size=64,
                 dropout=0.3):
        super(CNN, self).__init__()

        #* 1st convolutional layer
        out_shape = self._get_out_shape(height=input_shape[1], width=input_shape[2], kernel_size=kernel_size1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=kernel_size1)
        self.zero_pad1 = nn.ZeroPad2d(self._get_same_padding(height=out_shape[0],
                                                             width=out_shape[1],
                                                             kernel_size=kernel_size1,
                                                             stride=stride1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=kernel_size1, stride=stride1)
        self.batch_norm1 = nn.BatchNorm2d(hidden_channels)

        #* 2nd convolutional layer
        out_shape = self._get_out_shape(height=out_shape[0], width=out_shape[1], kernel_size=kernel_size2)
        self.conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=kernel_size2)
        self.zero_pad2 = nn.ZeroPad2d(self._get_same_padding(height=out_shape[0],
                                                             width=out_shape[1],
                                                             kernel_size=kernel_size2,
                                                             stride=stride2))
        self.max_pool2 = nn.MaxPool2d(kernel_size=kernel_size2, stride=stride2)
        self.batch_norm2 = nn.BatchNorm2d(hidden_channels)

        #* 3rd convolutional layer
        out_shape = self._get_out_shape(height=out_shape[0], width=out_shape[1], kernel_size=kernel_size3)
        self.conv3 = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(2, 2))
        self.zero_pad3 = nn.ZeroPad2d(self._get_same_padding(height=out_shape[0],
                                                             width=out_shape[1],
                                                             kernel_size=kernel_size3,
                                                             stride=stride3))
        self.max_pool3 = nn.MaxPool2d(kernel_size=kernel_size3, stride=stride3)
        self.batch_norm3 = nn.BatchNorm2d(hidden_channels)

        #* Other layers
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        #* Fully connected out layers
        self.fc1 = nn.Linear(in_features=hidden_channels * out_shape[0] * out_shape[1], out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x: torch.Tensor):

        #* Convolutional layers
        x = self.max_pool1(self.zero_pad1(self.conv1(x)))
        x = self.activation(self.batch_norm1(x))
        x = self.max_pool2(self.zero_pad2(self.conv2(x)))
        x = self.activation(self.batch_norm2(x))
        x = self.max_pool3(self.zero_pad3(self.conv3(x)))
        x = self.activation(self.batch_norm3(x))

        #* Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _get_out_shape(self, height, width, kernel_size, stride=(1, 1), height_padding=0, width_padding=0, dilation=(1, 1)):
        #* Make int to tuple of int if necessary
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation)

        out_height = (height + height_padding - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
        out_width = (width + width_padding - dilation[0] * (kernel_size[1] - 1) - 1) / stride[0] + 1
        return int(np.floor(out_height)), int(np.floor(out_width))

    def _get_same_padding(self, height, width, kernel_size, stride, dilation=(1, 1)):
        #* Make int to tuple of int if necessary
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation)

        height_padding = (height - 1) * stride[0] - height + dilation[0] * (kernel_size[0] - 1) + 1
        width_padding = (width - 1) * stride[1] - width + dilation[1] * (kernel_size[1] - 1) + 1

        left_padding = int(np.ceil(width_padding / 2))
        right_padding = int(np.floor(width_padding / 2))
        top_padding = int(np.ceil(height_padding / 2))
        bottom_padding = int(np.floor(height_padding / 2))
        return left_padding, right_padding, top_padding, bottom_padding


class generic_RNN(nn.Module):
    def __init__(self, rnn_type, input_shape, output_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(generic_RNN, self).__init__()

        #* Network components : RNN
        self.rnn = getattr(nn, rnn_type)(input_size=input_shape[-1],
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         batch_first=True,
                                         bidirectional=bidirectional)

        #* Network components : Output layer
        if bidirectional:
            self.fc1 = nn.Linear(in_features=2 * hidden_size, out_features=hidden_size)
        else:
            self.fc1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(self.activation(self.fc1(x[:, -1, :])))
        return self.fc2(x)


class RNN(generic_RNN):
    def __init__(self, input_shape, output_size=10,
                 hidden_size=64, num_layers=2,
                 dropout=0.3):
        super().__init__(rnn_type="RNN",
                         input_shape=input_shape,
                         output_size=output_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout)

class biRNN(generic_RNN):
    def __init__(self, input_shape, output_size=10,
                 hidden_size=64, num_layers=2,
                 dropout=0.3):
        super().__init__(rnn_type="RNN",
                         input_shape=input_shape,
                         output_size=output_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         bidirectional=True,
                         dropout=dropout)


class LSTM(generic_RNN):
    def __init__(self, input_shape, output_size=10,
                 hidden_size=64, num_layers=2,
                 dropout=0.3):
        super().__init__(rnn_type="LSTM",
                         input_shape=input_shape,
                         output_size=output_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout)

class biLSTM(generic_RNN):
    def __init__(self, input_shape, output_size=10,
                 hidden_size=64, num_layers=2,
                 dropout=0.3):
        super().__init__(rnn_type="LSTM",
                         input_shape=input_shape,
                         output_size=output_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         bidirectional=True,
                         dropout=dropout)

class GRU(generic_RNN):
    def __init__(self, input_shape, output_size=10,
                 hidden_size=64, num_layers=2,
                 dropout=0.3):
        super().__init__(rnn_type="GRU",
                         input_shape=input_shape,
                         output_size=output_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         dropout=dropout)

class biGRU(generic_RNN):
    def __init__(self, input_shape, output_size=10,
                 hidden_size=64, num_layers=2,
                 dropout=0.3):
        super().__init__(rnn_type="GRU",
                         input_shape=input_shape,
                         output_size=output_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         bidirectional=True,
                         dropout=dropout)

if __name__ == "__main__":
    print("This is module classifier")
