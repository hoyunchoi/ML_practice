import argparse
import os
import time
import typing
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from tqdm.notebook import tqdm
import torch.autograd.profiler as profiler
from sklearn.metrics import mean_squared_error

from RNN import RNN
from stock_data import stock_data
from early_stopping import early_stopping

"""
    Various functions for making stock price predicting machine

    ---------------------------------------------------------------------
    check_features(in_features, out_features)
    Args:
        in_features : Input abb_features for RNN
        out_features : Output abb_features for RNN
    Return:
        in_features : sorted in_fetures in order of data column name
        out_features : sorted out_fetures in order of data column name
    ---------------------------------------------------------------------
    model_summary(model, input_size, device=None)
    Args:
        model: NN model
        input_size : input size of input model without batch dimension
        device : device where input model is declared (optional)
    Return:
        None
    ---------------------------------------------------------------------
    benchmark(model, data, loss_func, optimizer, param, warmup, sort_by, save_path)
    Benchmark the model with torch.autograd.profiler
    Args:
        model : RNN model
        data : stock_data class for model
        loss_func : pytorch loss function for training
        optimizer : pytorch optimizer for updating model parameter
        warup : How many times to warmup before profiling
        sort_by : Sorted order ex)self_cpu_time_total, cuda_time_total,...
        save_path: path to save profiled text
    Return:
        None
    ---------------------------------------------------------------------
    train(model, data, param, max_epoch, loss_func, optimizer, min_val_loss, use_tqdm, verbose)
    Args:
        model : RNN model
        data : stock_data class for model
        param : hyperparameter
        max_epoch : number of epochs to train
        loss_func : pytorch loss function for training
        optimizer : pytorch optimizer for updating model parameter
        min_val_loss : when subsequent training, input the last minimum validation loss
        use_tqdm: Whether to use tqdm package for output
        verbose: When 2, print every log.
                 When 1, print only log about early stoppping.
                 When 0, do not print any log
    Return:
        train_loss_list : ndarray storing loss for training per each epoch
        val_loss_list : ndarray storing loss for validation per each epoch
                        (if val_loader is not given, it is zero array)
        (best_model, best_epoch): model state and epoch at minimum validation loss
    ---------------------------------------------------------------------
    plot_loss(train_loss_list, val_loss_list, ax, stop_epoch, save_path):
    Args:
        train_loss_list : list of training loss
        val_loss_list : list of validation loss
        ax : When given, draw plot at the ax
        stop_epoch : when given, mark the early stop point
        save_path : If given, save the plot to the save_path
    Return:
        ax: axes where figure is at
    ---------------------------------------------------------------------
    average_test(model, data, loss_func, successive_days=None, device=None):
    Args:
        model : (trained) NN model
        data : class stock_data
        loss_func : pytorch loss function used at training
        successive_days : Number of successive days for machine to predict
        device : device where input model is declared (optional)
    Return:
        prediction_list : data that model has predicted
        test_loss : loss of test data
    ---------------------------------------------------------------------
    plot_prediction(data, epoch, avg_prediction, save_path=None):
    Args:
        data : class stock_data
        epoch : How many epoch the model is trained
        avg_prediction : data that model has predicted
        save_path : If given, save the plot to the save_path
    Return:
        ax
    ---------------------------------------------------------------------
    pre_processed_name(stocks, fmt='parquet', comp='snappy'):
    Args:
        stocks : Name of stocks
        fmt : format for saved data (default: 'parquet')
        comp : compression method for saved data (default: 'snappy')
    Return:
        Relative path of pre-processed data. EX) 'preprocessed_data/IBM.parquet.snappy'
    ---------------------------------------------------------------------
"""

FEATURE_ORDER = {'Open': 0, 'High': 1, 'Low': 2, 'Close': 3}

file_path_list = [file.path for file in os.scandir('data') if file.is_file()]
stocks_list = [file_path[file_path.find("/") + 1:file_path.find("_")] for file_path in file_path_list]
STOCKS_LIST = sorted(set(stocks_list) - set(['all']))


def check_features(in_features: typing.List[str], out_features: typing.List[str]):
    try:
        in_features.sort(key=lambda feature: FEATURE_ORDER[feature])
        out_features.sort(key=lambda feature: FEATURE_ORDER[feature])
    except KeyError:
        print("Defined in/out abb_features are invalid")
    return in_features, out_features


def model_summary(model: RNN,
                  input_size: int,
                  precision: str = '32'):
    summary(model,
            input_size=input_size,
            batch_dim=0,
            dtypes=[getattr(torch, 'float' + precision)],
            device=next(model.parameters()).device,
            verbose=2,
            col_names=[
                "input_size",
                "output_size",
                "kernel_size",
                "num_params",
                "mult_adds"],
            col_width=16)


def benchmark(model: RNN,
              data: stock_data,
              loss_func: torch.nn.modules.loss,
              optimizer: torch.optim,
              param: argparse.Namespace,
              warmup: int = 4,
              sort_by: str = 'self_cpu_time_total',
              save_path: str = 'profile.txt'
              ):
    #* Use the same device where NN is at
    device = model.device

    #* Prepare to train
    train_loader = data.get_train_loader(batch_size=param.train_batch)
    model.train()

    #* Warm-Up
    for _ in range(warmup):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            prediction = output.detach()
            for _ in range(param.successive_days - 1):
                prediction = model.forward_wo_initH(prediction.detach())
                output = torch.cat((output, prediction), axis=1)
            loss = loss_func(output, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    #* Profiling
    train_loss = 0.0
    for x, y in train_loader:
        with profiler.profile(profile_memory=True,
                              with_stack=True,
                              use_cuda=True
                              ) as prof:
            #* Profiling feed forward process
            with profiler.record_function("Feed Forward"):
                x, y = x.to(device), y.to(device)
                output = model(x)
                prediction = output.detach()
                for _ in range(param.successive_days - 1):
                    prediction = model.forward_wo_initH(prediction.detach())
                    output = torch.cat((output, prediction), axis=1)

            #* Profiling loss calculation process
            with profiler.record_function("Get Loss"):
                loss = loss_func(output, y)
                train_loss += loss.item()

            #* Profiling back propagation process
            with profiler.record_function("Back Propagation"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    #* Save the result of profiling
    prof.export_chrome_trace(save_path + ".json")
    with open(save_path + ".txt", 'w') as f:
        f.write("Sort by {}".format(sort_by))
        f.write(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_by, row_limit=10))


def train(model: RNN,
          data: stock_data,
          param: argparse.Namespace,
          max_epoch: int,
          loss_func: torch.nn.modules.loss,
          optimizer: torch.optim,
          min_val_loss: float = np.Inf,
          use_tqdm: bool = True,
          verbose: int = 1
          ) -> typing.Tuple[np.ndarray, np.ndarray, collections.OrderedDict, int]:

    start_time = time.time()

    #* Use the same device where NN is at
    device = model.device

    #* Tensor for storing loss per each epoch
    train_loss_list = torch.zeros(max_epoch, device=device)
    val_loss_list = torch.zeros(max_epoch, device=device)

    #* Whether to use tqdm
    if use_tqdm:
        epoch_range = tqdm(range(max_epoch), desc="Epoch", colour='green')
        trace_func = tqdm.write
    else:
        epoch_range = range(max_epoch)
        trace_func = print

    #* Initialize early stopping object
    if param.early_stop_patience:
        best_model = None
        best_epoch = -1
        es = early_stopping(verbose=verbose,
                            patience=param.early_stop_patience,
                            min_val_loss=min_val_loss,
                            delta=param.early_stop_delta,
                            trace_func=trace_func)

    #* Generate data loader
    train_loader = data.get_train_loader(batch_size=param.train_batch)
    val_loader = data.get_val_loader(batch_size=param.val_batch)

    for epoch in epoch_range:
        #* Training
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            #* Get input, output value
            x, y = x.to(device), y.to(device)

            #* Feed forwarding
            output = model(x)
            prediction = output.detach()
            for _ in range(param.successive_days - 1):
                prediction = model.forward_wo_initH(prediction.detach())
                output = torch.cat((output, prediction), axis=1)
            loss = loss_func(output, y)

            #* Back propagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            #* Calculate loss
            train_loss += loss.item()

        #* End of a epoch. Store average loss of current epoch
        train_loss /= len(data.train_dataset)
        train_loss_list[epoch] = train_loss

        #* Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                #* Get input, output value
                x, y = x.to(device), y.to(device)

                #* Feed forwarding
                output = model(x)
                prediction = output.detach()
                for _ in range(param.successive_days - 1):
                    prediction = model.forward_wo_initH(prediction.detach())
                    output = torch.cat((output, prediction), axis=1)

                #* Calculate loss
                loss = loss_func(output, y)
                val_loss += loss.item()

        #* End of a epoch. Store average loss of current epoch
        val_loss /= len(data.val_dataset)
        val_loss_list[epoch] = val_loss

        #* Early stopping
        if param.early_stop_patience:
            if es(epoch + 1, val_loss):
                best_epoch = epoch
                best_model = model.state_dict()
            if es.early_stop:
                if verbose:
                    trace_func("Early Stopping!")
                break

        #* Print the result
        if verbose > 1:
            trace_func("Epoch {}/{}".format(epoch + 1, max_epoch), end='\t')
            trace_func("train loss: {:.6f}\t validation loss: {:.6f}".format(train_loss, val_loss))

    if not param.early_stop_patience:
        best_model = model.state_dict()
        best_epoch = max_epoch - 1
        min_val_loss = val_loss_list[-1]
    else:
        min_val_loss = es.min_val_loss

    print("Train finished with {:.6f} seconds, {} epochs, {:.6f} validation loss".format(time.time() - start_time, epoch, min_val_loss))
    return train_loss_list.cpu().numpy(), val_loss_list.cpu().numpy(), (best_model, best_epoch)


def plot_loss(train_loss_list: np.ndarray,
              val_loss_list: np.ndarray,
              ax=None,
              stop_epoch=None,
              save_path=None):
    assert train_loss_list.shape == val_loss_list.shape, "train loss and validation loss should have same length"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(np.arange(1, len(train_loss_list) + 1, 1), train_loss_list, 'bo-', label='train')

    #* Plot only when val_loss_list is not zero array
    if np.count_nonzero(val_loss_list):
        ax.plot(np.arange(1, len(val_loss_list) + 1, 1), val_loss_list, 'go-', label='validation')

    if stop_epoch is not None:
        ax.plot([stop_epoch, stop_epoch], [0, max(np.max(train_loss_list), np.max(val_loss_list))], 'r--', label="Early Stop")

    ax.set_xlabel("Epoch", fontsize=30)
    ax.set_ylabel("loss", fontsize=30)
    ax.legend(loc='best', fontsize=30, frameon=False)
    ax.tick_params(axis='both', labelsize=20, direction='in')
    ax.set_xlim([0, len(train_loss_list) + 1])

    try:
        if save_path:
            fig.savefig(save_path, facecolor='w')
        else:
            fig.show()
    except UnboundLocalError:
        pass
    return ax


def average_test(model: RNN,
                 data: stock_data,
                 verbose: bool = False
                 ) -> typing.Tuple[np.ndarray, float]:
    start_time = time.time()

    #* Get information from model
    device = model.device
    successive_days = data.successive_days

    #* ndarray for storing prediction of machine
    prediction_list = torch.zeros((len(data.test_raw) - data.past_days, len(data.out_features)), device=device)
    prediction_num = np.ones(len(data.test_raw) - data.past_days) * successive_days
    for sd in range(1, successive_days):
        prediction_num[:successive_days - sd] -= 1
        prediction_num[-successive_days + sd:] -= 1

    #* Test loader : batch size of 1
    test_loader = data.get_test_loader()

    #* Test the model
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            #* Feed forwarding
            output = model(x)
            prediction = output.detach()
            for _ in range(data.successive_days - 1):
                prediction = model.forward_wo_initH(prediction.detach())
                output = torch.cat((output, prediction), axis=1)

            #* Save all prediction to prediction_list
            prediction_list[i:i + successive_days, :] += output.squeeze(0)

    #* Average prediction list
    prediction_list = prediction_list.cpu().numpy() / prediction_num[:, np.newaxis]

    #* Rescale the predicted data
    prediction_list = data.test_output_scaler.inverse_transform(prediction_list)

    #* Get MSE Loss
    real_out = data.data_frame[data.val_test_boundary:][data.out_features]
    try:
        mse = mean_squared_error(y_true=real_out, y_pred=prediction_list)
        mse /= len(data.out_features) * len(prediction_list)
    except ValueError:
        mse = np.nan

    #* Print the result
    if verbose:
        print("Average test finished with {:.2f} seconds with MSE {:.6f}".format(time.time() - start_time, mse))
    return prediction_list, mse


def plot_prediction(data: stock_data,
                    epoch: int,
                    avg_prediction: np.ndarray = None,
                    save_path: str = None):
    num_features = len(data.out_features)
    fig = plt.figure(figsize=(10 * num_features, 10))

    for i in range(num_features):
        ax = plt.subplot(1, num_features, i + 1)
        real_out = data.data_frame[data.val_test_boundary:][data.out_features[i]]
        ax.plot(real_out, 'r-', label='Real')

        #* Plot average prediction
        if avg_prediction is not None:
            ax.plot(real_out.index[:], avg_prediction[:, i], 'b-', label='Average predicted')

        ax.set_xlabel("Time", fontsize=25)
        ax.set_ylabel("Price", fontsize=25)
        ax.legend(loc='best', fontsize=25, frameon=False)
        ax.tick_params(axis='both', labelsize=20, direction='in')
        ax.set_title(data.out_features[i], fontsize=30)
    fig.autofmt_xdate()
    fig.suptitle("Test result after {} epochs".format(epoch), fontsize=35)
    if save_path:
        fig.savefig(save_path, facecolor='w')
    else:
        fig.show()
    return ax


def pre_processed_name(stocks: str,
                       fmt: str = 'parquet',
                       comp: str = 'snappy') -> str:
    return '.'.join([stocks, fmt, comp])


if __name__ == "__main__":
    print("This is module utils")
