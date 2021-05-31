import utils
from utils import STOCKS_LIST, FEATURE_ORDER
import torch
import torch.optim as optim
import torch.nn as nn
from stock_data import stock_data
from RNN import RNN
import time


class default_parameter():
    #* Set default values
    def __init__(self):
        #* Basic parameters
        self.precision = '32'

        #* Parameters for data input
        self.stocks = "AMZN"
        self.feature_abbreviation = {'O': 'Open',
                                     'C': 'Close',
                                     'H': 'High',
                                     'L': 'Low'}
        self.in_features = ['Open', 'Close', 'High', 'Low']
        self.out_features = ['High', 'Low']

        #* Parameters for processed data
        self.past_days = 80  # ! 60(2) -> 80(9)
        self.successive_days = 1
        self.scaler_name = 'MinMaxScaler'  # ! MinMaxScaler(1) -> MinMaxScaler(9)

        #* Parameters for NN
        self.rnn_name = 'LSTM'  # ! GRU(1) -> LSTM(6)
        self._update_rnn_type()
        self.num_layers = 3  # ! 3(3) -> 10
        self.hidden_size = 200  # ! 200(3) -> 10
        self.dropout = 0.1  # ! 0.0(4) -> 0.1(8)

        #* Parameters for training
        self.optimizer_name = 'RMSprop'  # ! Adam(5) -> RMSprop(7)
        self.learning_rate = 5e-4  # ! 1e-3(5) -> 5e-4(7)
        self.L2_regularization = 1e-5  # ! 1e-4(4) -> 1e-5(8)
        self.loss_name = 'SmoothL1Loss'  # ! L1Loss(2) -> SmoothL1Loss(6)

        #* Parameters for early stopping
        self.early_stop_patience = 10
        self.early_stop_delta = 0

        #* Get prefix of parameters
        self.prefix = ''
        self.update_prefix()

    def update_prefix(self):
        #* Basic information
        self.prefix = '_'.join([self.stocks, self.precision, self.rnn_name])

        #* Features
        self.prefix += '_' + ''.join([feature[0] for feature in self.in_features]) + '2' + ''.join([feature[0] for feature in self.out_features])

        #* Past days and successive days
        self.prefix += '_'.join(['',
                                 'PD{}'.format(self.past_days),
                                 'SD{}'.format(self.successive_days)])

        #* scaler
        self.prefix += '_' + self.scaler_name[:self.scaler_name.find("Scaler")]

        #* NN hyper parameters
        self.prefix += '_'.join(['',
                                 'NL{}'.format(self.num_layers),
                                 'HL{}'.format(self.hidden_size),
                                 'DO{}'.format(self.dropout)])

        #* Train hyper parameters
        self.prefix += '_'.join(['',
                                 self.optimizer_name,
                                 'LR{}'.format(self.learning_rate),
                                 'RG{}'.format(self.L2_regularization),
                                 self.loss_name])

    def read_prefix(self, prefix: str):
        param_list = prefix.split('_')

        #* Basic information
        self.stocks = param_list[0]
        self.precision = param_list[1]
        self.rnn_name = param_list[2]
        self._update_rnn_type()

        #* Features
        feature = param_list[3].split('2')
        self.in_features, self.out_features = [], []
        for in_feature in feature[0]:
            self.in_features.append(self.feature_abbreviation[in_feature])
        for out_feature in feature[1]:
            self.out_features.append(self.feature_abbreviation[out_feature])

        #* Past days and successive days
        self.past_days = param_list[4][2:]
        self.successive_days = param_list[5][2:]

        #* scaler
        self.scaler_name = param_list[6] + 'Scaler'

        #* NN hyper parameters
        self.num_layers = param_list[7][2:]
        self.hidden_size = param_list[8][2:]
        self.dropout = param_list[9][2:]

        #* Train hyper parameters
        self.optimizer_name = param_list[10]
        self.learning_rate = param_list[11][2:]
        self.L2_regularization = param_list[12][2:]
        self.loss_name = param_list[13]

        #* Update refix
        self.prefix = prefix

    def set_parameter(self, name, value):
        if name == 'stocks':
            self._check_stocks(value)
        elif 'features' in name:
            value = self._check_features(value)
        elif 'name' in name:
            assert isinstance(value, str), "Expected string for {}".format(name)
        elif ('days' in name or
              'num_layers' == name or
              'hidden_size' == name or
              'early_stop_patience' == name):
            assert isinstance(value, int), "Expected int for {}".format(name)
        else:
            assert isinstance(value, float), 'Expected float for {}'.format(name)
        setattr(self, name, value)
        if name == 'rnn_name':
            self._update_rnn_type()
        elif name == 'rnn_type' or name == 'bidirectional':
            self._update_rnn_name()

    def _check_stocks(self, stocks):
        if stocks not in STOCKS_LIST:
            raise Exception("Given stocks is not valid")

    def _check_features(self, features):
        try:
            features.sort(key=lambda feature: FEATURE_ORDER[feature])
        except KeyError:
            print("Defined in/out abb_features are invalid")
        return features

    def _update_rnn_type(self):
        if 'bi' in self.rnn_name:
            self.bidirectional = True
            self.rnn_type = self.rnn_name[2:]
        else:
            self.bidirectional = False
            self.rnn_type = self.rnn_name

    def _update_rnn_name(self):
        if self.bidirectional:
            self.rnn_name = 'bi' + self.rnn_type
        else:
            self.rnn_name = self.rnn_type


class tuner():
    def __init__(self, parameter: default_parameter, device: torch.device):
        #* Parameters for data input
        self.stocks = parameter.stocks
        self.in_features = parameter.in_features
        self.out_features = parameter.out_features

        #* Parameters for processed data
        self.past_days = parameter.past_days
        self.successive_days = parameter.successive_days
        self.scaler_name = parameter.scaler_name

        #* Parameters for NN
        if "bi" in parameter.rnn_name:
            self.rnn_type = parameter.rnn_name[2:]
            self.bidirectional = True
        else:
            self.rnn_type = parameter.rnn_name
            self.bidirectional = False
        self.num_layers = parameter.num_layers
        self.hidden_size = parameter.hidden_size
        self.dropout = parameter.dropout

        #* Parameters for training
        self.optimizer_name = parameter.optimizer_name
        self.learning_rate = parameter.learning_rate
        self.L2_regularization = parameter.L2_regularization
        self.loss_name = parameter.loss_name

        #* Parameters for early stopping
        self.early_stop_patience = parameter.early_stop_patience
        self.early_stop_delta = parameter.early_stop_delta

        #* Specifiy device and declare 'data', 'model', 'optimizer', 'loss_func'
        self.device = device
        self.data: stock_data = None
        self.model: RNN = None
        self.optimizer: optim = None
        self.loss_func: nn = None

        #* Dictionary to store every results
        self.result = {}

    #* Run the tuner
    def run(self):
        self._prepare_data()
        self._prepare_model()
        self._train()
        self._test()

    #* Prepare to train neural network
    def _prepare_data(self):
        self.data = stock_data(stocks=self.stocks,
                               in_features=self.in_features,
                               out_features=self.out_features,
                               past_days=self.past_days,
                               successive_days=self.successive_days,
                               scaler_name=self.scaler_name,
                               verbose=False)
        self.train_loader = self.data.get_train_loader(batch_size=64)
        self.val_loader = self.data.get_val_loader(batch_size=64)

    def _prepare_model(self):
        self.model = RNN(rnn_type=self.rnn_type,
                         in_features=self.in_features,
                         out_features=self.out_features,
                         successive_days=self.successive_days,
                         bidirectional=self.bidirectional,
                         num_layers=self.num_layers,
                         hidden_size=self.hidden_size,
                         dropout=self.dropout).to(self.device)
        self.optimizer = getattr(optim, self.optimizer_name)(params=self.model.parameters(),
                                                             lr=self.learning_rate,
                                                             weight_decay=self.L2_regularization)
        self.loss_func = getattr(nn, self.loss_name)(reduction='sum')
        self.epoch = 0

    #* Train the model
    def _train(self):
        start_time = time.time()
        train_loss_list, val_loss_list, best = utils.train(model=self.model,
                                                           max_epoch=1000,  # Large enough to get early stop
                                                           loss_func=self.loss_func,
                                                           optimizer=self.optimizer,
                                                           train_loader=self.train_loader,
                                                           val_loader=self.val_loader,
                                                           early_stop_patience=self.early_stop_patience,
                                                           early_stop_delta=self.early_stop_delta,
                                                           use_tqdm=False,
                                                           verbose=0)
        best_epoch = best[1]
        train_loss_list = train_loss_list[:best_epoch + 1 + self.early_stop_patience]
        val_loss_list = val_loss_list[:best_epoch + 1 + self.early_stop_patience]
        self.result['time'] = time.time() - start_time
        self.result['train_loss_list'] = train_loss_list
        self.result['val_loss_list'] = val_loss_list
        self.result['epoch'] = best_epoch

    #* Test the model
    def _test(self):
        avg_prediction, avg_test_loss, avg_test_mse = utils.average_test(model=self.model,
                                                                         data=self.data,
                                                                         loss_func=self.loss_func,
                                                                         verbose=False)
        recurrent_prediction, recurrent_test_loss, recurrent_test_mse = utils.recurrent_test(model=self.model,
                                                                                             data=self.data,
                                                                                             loss_func=self.loss_func,
                                                                                             verbose=False)
        self.result['avg_test_loss'] = avg_test_loss
        self.result['avg_prediction'] = avg_prediction
        self.result['avg_test_mse'] = avg_test_mse
        self.result['recurrent_test_loss'] = recurrent_test_loss
        self.result['recurrent_prediction'] = recurrent_prediction
        self.result['recurrent_test_mse'] = recurrent_test_mse


if __name__ == "__main__":
    param = default_parameter()
    param.read_prefix('AMZN_32_LSTM_OCHL2HL_PD80_SD1_MinMax_NL3_HL200_DO0.1_RMSprop_LR0.0005_RG1e-05_SmoothL1Loss')
    print(param.prefix)
    # print(param.in_features, param.out_features)
    # print("This is module tuning")
