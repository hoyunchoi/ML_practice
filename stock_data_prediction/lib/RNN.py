import argparse
import torch
import torch.nn as nn

"""
    class RNN

    (public) Members
    num_in_features : Number of input features of RNN
    num_out_features : Number of output features of RNN
    successive_days : Number of days to predict

    (public) Methods
    forward : basic pytorch nn forward
    forward_wo_initH: Do forward without initializing hidden (cell) state
                      Use when you need to keep gradient flow
"""

#* Generate RNN
class RNN(nn.Module):
    def __init__(self, param: argparse.Namespace):
        super(RNN, self).__init__()

        #* Get input variables
        self.num_in_features = len(param.in_features)
        self.num_out_features = len(param.out_features)
        self.hidden_size = param.hidden_size
        self.num_layers = param.num_layers

        self.dropout = param.dropout
        self.rnn_dropout = 0.0 if self.num_layers == 1 else self.dropout
        self.use_bn = param.use_bn
        self.device = param.device

        #* Get the type of rnn
        if 'bi' in param.rnn_name:
            self.bidirectional = True
            self.num_directions = 2
            self.rnn_type = param.rnn_name[2:]
        else:
            self.bidirectional = False
            self.num_directions = 1
            self.rnn_type = param.rnn_name

        #* Initializing hidden (cell) state
        if self.rnn_type == 'LSTM':
            self.init_hidden = self._init_hc
            self.hidden = torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        else:
            self.init_hidden = self._init_h
            self.hidden = torch.tensor([], device=self.device)

        #* Network components : RNN
        self.rnn = getattr(nn, self.rnn_type)(input_size=self.num_in_features,
                                              hidden_size=self.hidden_size,
                                              num_layers=self.num_layers,
                                              dropout=self.rnn_dropout,
                                              batch_first=True,
                                              bidirectional=self.bidirectional)

        #* Network components : Output layer
        self.out_layer = self._generate_out_layer()

    #* Initializing hidden and cell state : Used for LSTM type network
    def _init_hc(self, x):
        h0 = torch.zeros(self.num_directions * self.num_layers, x.shape[0], self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_directions * self.num_layers, x.shape[0], self.hidden_size, device=self.device)
        return h0, c0

    #* Initializing hidden state : Used for RNN/GRU type network
    def _init_h(self, x):
        h0 = torch.zeros(self.num_directions * self.num_layers, x.shape[0], self.hidden_size, device=self.device)
        return h0

    #* Forward method
    def forward(self, x):
        hidden0 = self.init_hidden(x)
        x, self.hidden = self.rnn(x, hidden0)
        x = self.out_layer(x[:, -1, :])
        return x.unsqueeze(1)

    #* Do forward without initializing hidden (cell) state: When you need to keep gradient flow
    def forward_wo_initH(self, x):
        x, self.hidden = self.rnn(x, self.hidden)
        x = self.out_layer(x[:, -1, :])
        return x.unsqueeze(1)

    def _generate_out_layer(self):
        layers = []
        #* Batch normalization
        if self.use_bn:
            layers.append(nn.BatchNorm1d(self.num_directions * self.hidden_size))

        #* Dropout
        layers.append(nn.Dropout(self.dropout))

        #* Fully connected layer
        fc = nn.Linear(in_features=self.num_directions * self.hidden_size,
                       out_features=self.num_out_features)
        nn.init.xavier_uniform_(fc.weight)
        layers.append(fc)
        return nn.Sequential(*layers)

if __name__ == "__main__":
    print("This is module RNN")
