import utils
import torch
import torch.optim as optim
import torch.nn as nn
from stock_data import stock_data
from RNN import RNN
import time
import argparse


FEATURE_ABBREVIATION = {'O': 'Open', 'C': 'Close', 'H': 'High', 'L': 'Low'}


def set_default() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    args = parser.parse_args('')

    #* Basic parameters
    args.precision = '32'
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #* Parameters for data input
    args.stocks = "AMZN"
    args.stocks_list = ['AMZN']
    args.in_features = ['High', 'Low']
    args.out_features = ['High', 'Low']

    #* Parameters for processed data
    #? 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    args.past_days = 30                                     # ! 30(2)
    args.successive_days = 5
    #? 'MinMaxScaler', 'StandardScaler'
    args.scaler_name = 'StandardScaler'                     # ! StandardScaler(2)

    #* Parameters for NN
    #? 'RNN', 'LSTM', 'GRU', 'biRNN', 'biLSTM', 'biGRU'
    args.rnn_name = 'GRU'                                   # ! GRU(1)
    #? 1,2,3,4,5
    args.num_layers = 1                                     # ! 1(3)
    #? 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
    args.hidden_size = 150                                  # ! 150(3)
    #? 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
    args.dropout = 0.2                                      # ! 0.2(5)
    #? True, False
    args.use_bn = False                                     # ! False(4)

    #* Parameters for training
    args.train_batch = 512
    args.val_batch = 512
    #? 'Adam', 'RMSprop', 'SGD', 'Adagrad', 'Adadelta'
    args.optimizer_name = 'Adam'                            # ! Adam(6)
    #? 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2
    args.learning_rate = 0.005                              # ! 0.005(6)
    #? 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2
    args.L2 = 0.05                                          # ! 0.05(4,5)
    #? 'L1Loss', 'SmoothL1Loss', 'MSELoss'
    args.loss_name = 'MSELoss'                              # ! MSELoss(1)

    #* Parameters for early stopping
    args.early_stop_patience = 10
    args.early_stop_delta = 0
    return args


def generate_prefix(args: argparse.Namespace) -> str:
    #* Basic information
    prefix = '_'.join([args.stocks, args.precision, args.rnn_name])

    #* Features
    prefix += '_' + ''.join([feature[0] for feature in args.in_features]) + '2' + ''.join([feature[0] for feature in args.out_features])

    #* Past days and successive days
    prefix += '_'.join(['',
                        'PD{}'.format(args.past_days),
                        'SD{}'.format(args.successive_days)])

    #* scaler
    prefix += '_' + args.scaler_name[:args.scaler_name.find("Scaler")]

    #* NN hyper parameters
    prefix += '_'.join(['',
                        'NL{}'.format(args.num_layers),
                        'HL{}'.format(args.hidden_size),
                        'DO{}'.format(args.dropout)])
    prefix += '_BN1' if args.use_bn else '_BN0'

    #* Train hyper parameters
    prefix += '_'.join(['',
                        args.optimizer_name,
                        'LR{}'.format(args.learning_rate),
                        'RG{}'.format(args.L2),
                        args.loss_name])
    return prefix


def read_prefix(prefix: str) -> argparse.Namespace:
    #* Default param
    args = set_default()
    param_list = prefix.split('_')

    #* Basic information
    args.stocks = param_list[0]
    args.stocks_list = [args.stocks]
    args.precision = param_list[1]
    args.rnn_name = param_list[2]

    #* Features
    feature = param_list[3].split('2')
    args.in_features, args.out_features = [], []
    for in_feature in feature[0]:
        args.in_features.append(FEATURE_ABBREVIATION[in_feature])
    for out_feature in feature[1]:
        args.out_features.append(FEATURE_ABBREVIATION[out_feature])

    #* Past days and successive days
    args.past_days = int(param_list[4][2:])
    args.successive_days = int(param_list[5][2:])

    #* scaler
    args.scaler_name = param_list[6] + 'Scaler'

    #* NN hyper parameters
    args.num_layers = int(param_list[7][2:])
    args.hidden_size = int(param_list[8][2:])
    args.dropout = float(param_list[9][2:])
    args.bn = bool(param_list[10][2:])

    #* Train hyper parameters
    args.optimizer_name = param_list[11]
    args.learning_rate = float(param_list[12][2:])
    args.L2 = float(param_list[13][2:])
    args.loss_name = param_list[14]

    return args


class tuner():
    def __init__(self, param: argparse.Namespace):
        #* Save hyper parameters
        self.param = param

        #* Specifiy device and declare 'data', 'model', 'optimizer', 'loss_func'
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
        self.data = stock_data(param=self.param,
                               pre_process=True,
                               use_dataset=True,
                               verbose=False)

    def _prepare_model(self):
        self.model = RNN(param=self.param).to(self.param.device)
        self.optimizer = getattr(optim, self.param.optimizer_name)(params=self.model.parameters(),
                                                                   lr=self.param.learning_rate,
                                                                   weight_decay=self.param.L2)
        self.loss_func = getattr(nn, self.param.loss_name)(reduction='sum').to(self.param.device)
        self.epoch = 0

    #* Train the model
    def _train(self):
        start_time = time.time()
        train_loss, val_loss, best = utils.train(model=self.model,
                                                 data=self.data,
                                                 max_epoch=1000,  # Large enough to get early stop
                                                 loss_func=self.loss_func,
                                                 optimizer=self.optimizer,
                                                 param=self.param,
                                                 use_tqdm=0,
                                                 verbose=0)

        #* Cut only before best epoch
        best_epoch = best[1]
        train_loss = train_loss[:best_epoch + 1 + self.param.early_stop_patience]
        val_loss = val_loss[:best_epoch + 1 + self.param.early_stop_patience]
        self.result['time'] = (time.time() - start_time) / len(train_loss)
        self.result['train_loss'] = train_loss
        self.result['val_loss'] = val_loss
        self.result['epoch'] = best_epoch

    #* Test the model
    def _test(self):
        avg_prediction, avg_test_mse = utils.average_test(self.model, self.data, verbose=False)
        self.result['avg_prediction'] = avg_prediction
        self.result['avg_test_mse'] = avg_test_mse

if __name__ == "__main__":
    print("This is module tuning")
