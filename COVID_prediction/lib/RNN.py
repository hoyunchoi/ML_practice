import torch.nn as nn
import typing
"""
    class RNN

    (public) Members
    num_in_features : Number of input features of RNN
    num_out_features : Number of output features of RNN
    successive_days : Number of days to predict

    (public) Methods
    forward : basic pytorch nn forward
"""

#* Generate RNN


class RNN(nn.Module):
    def __init__(self, rnn_type: str,
                 in_features: typing.List[str],
                 out_features: typing.List[str] = None, successive_days=1, hidden_size=50, num_layers=4, dropout=0.2, bidirectional=False):
        super(RNN, self).__init__()

        #* If out features are given, return it. Else, out feature is same as in features
        self.num_in_features = len(in_features)
        self.num_out_features = len(out_features) if out_features else self.num_in_features
        self.successive_days = successive_days

        #* Get the type of rnn
        self.rnn_type = rnn_type

        #* Network components : RNN
        self.rnn = getattr(nn, self.rnn_type)(input_size=self.num_in_features,
                                              hidden_size=hidden_size,
                                              num_layers=num_layers,
                                              dropout=dropout,
                                              batch_first=True,
                                              bidirectional=bidirectional)
        #* Network components : Output layer
        if bidirectional:
            self.fc = nn.Linear(in_features=2 * hidden_size, out_features=self.num_out_features * self.successive_days)
        else:
            self.fc = nn.Linear(in_features=hidden_size, out_features=self.num_out_features * self.successive_days)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x.view(-1, self.successive_days, self.num_out_features)


if __name__ == "__main__":
    print("This is module RNN")
