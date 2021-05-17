import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm.notebook import tqdm

from RNN import RNN
from stock_data import stock_data

"""
    Various functions for making stock price predicting machine

    ---------------------------------------------------------------------
    check_features(in_features, out_features)
    Args:
        in_features : Input features for RNN
        out_features : Output features for RNN
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
    train(model, max_epoch, loss_func, optimizer, train_loader, val_loader=None, device=None)
    Args:
        model : NN model
        max_epoch : number of epochs to train
        loss_func : pytorch loss function for training
        optimizer : pytorch optimizer for updating model parameter
        train_loader : pytorch data loader for training
        val_loader : pytorch data loader for validation (optional)
        device : device where input model is declared (optional)
    Return:
        train_loss_list : ndarray storing loss for training per each epoch
        val_loss_list : ndarray storing loss for validation per each epoch
                        (if val_loader is not given, it is zero array)
    ---------------------------------------------------------------------
    plot_loss(train_loss_list, val_loss_list):
    Args:
        train_loss_list : list of training loss
        val_loss_list : list of validation loss
    Return:
        None
    ---------------------------------------------------------------------
    plot_loss(train_loss_list, val_loss_list, save_path=None):
    Args:
        train_loss_list : list of training loss
        val_loss_list : list of validation loss
        save_path : If given, save the plot to the save_path
    ---------------------------------------------------------------------
    test(model, data, loss_func, successive_days=None, device=None):
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
    plot_prediction(prediction_list, data, save_path=None):
    Args:
        prediction_list : data that model has predicted
        data : class stock_data
        save_path : If given, save the plot to the save_path
    Return:
        None
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


def check_features(in_features: list, out_features: list):
    try:
        in_features.sort(key=lambda feature: FEATURE_ORDER[feature])
        out_features.sort(key=lambda feature: FEATURE_ORDER[feature])
    except KeyError:
        print("Defined in/out features are invalid")
    return in_features, out_features


def model_summary(model: RNN, input_size, precision='32'):
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


def train(model: RNN,
          max_epoch,
          loss_func: torch.nn.modules.loss,
          optimizer: torch.optim,
          train_loader: torch.utils.data.dataloader.DataLoader,
          val_loader: torch.utils.data.dataloader.DataLoader = None,
          verbose=False):
    start_time = time.time()

    #* Use the same device where NN is at
    device = next(model.parameters()).device

    # * ndarray for storing loss per each epoch
    train_loss_list = np.zeros(max_epoch)
    val_loss_list = np.zeros(max_epoch)

    # * Train and validate(optional)
    for epoch in tqdm(range(max_epoch), desc="Epoch", colour='green'):
        model.train()
        train_loss = 0.0
        train_num = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]

            #* Feed forwarding
            output = model(x)
            loss = loss_func(output, y)

            #* Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #* Calculate loss
            train_loss += loss.item() * batch_size
            train_num += batch_size

        #* End of a epoch. Store average loss of current epoch
        train_loss /= train_num
        train_loss_list[epoch] = train_loss

        #* Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_num = 0
            with torch.no_grad():
                for x, y in val_loader:
                    #* Get input, output value
                    x, y = x.to(device), y.to(device)
                    batch_size = x.shape[0]

                    #* Feed forwarding
                    output = model(x)
                    loss = loss_func(output, y)

                    #* Calculate loss
                    val_loss += loss.item() * batch_size
                    val_num += batch_size
            #* End of a epoch. Store average loss of current epoch
            val_loss /= val_num
            val_loss_list[epoch] = val_loss

        if verbose:
            tqdm.write("Epoch {}/{}\t train loss: {:.6f}\t validation loss: {:.6f}".format(epoch + 1, max_epoch, train_loss, val_loss))
    print("Train finished with {:.6f} seconds".format(time.time() - start_time))
    return train_loss_list, val_loss_list


def plot_loss(train_loss_list, val_loss_list, save_path=None):
    assert train_loss_list.shape == val_loss_list.shape, "train loss and validation loss should have same length"

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(train_loss_list, 'bo-', label='train')

    # * Plot only when val_loss_list is not zero array
    if np.count_nonzero(val_loss_list):
        ax.plot(val_loss_list, 'go-', label='validation')

    ax.set_xlabel("Epoch", fontsize=30)
    ax.set_ylabel("loss", fontsize=30)
    ax.legend(loc='best', fontsize=30, frameon=False)
    ax.tick_params(axis='both', labelsize=20, direction='in')

    if save_path:
        fig.savefig(save_path, facecolor='w')
    else:
        fig.show()


def average_test(model: RNN,
                 data: stock_data,
                 loss_func: torch.nn.modules.loss = None,
                 verbose=False):
    start_time = time.time()

    #* Get information from model
    device = next(model.parameters()).device
    successive_days = model.successive_days

    # * ndarray for storing prediction of machine
    prediction_list = np.zeros((len(data.test_raw) - data.past_len, len(data.out_features)))
    prediction_num = np.zeros(len(data.test_raw) - data.past_len)

    #* Test loader : batch size of 1
    test_loader = data.get_test_loader()

    #* Test the model
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            #* Do prediction
            x, y = x.to(device), y.to(device)
            prediction = model(x)

            #* Save all prediction to prediction_list
            prediction_list[i:i + successive_days, :] += prediction.squeeze(0).cpu().numpy()
            prediction_num[i:i + successive_days] += 1

            #* Calculatet loss
            if loss_func is not None:
                loss = loss_func(prediction, y)
                test_loss += loss.item()

    #* Average loss and prediction list
    test_loss /= i + 1
    prediction_list /= prediction_num[:, np.newaxis]

    #* Print the result
    if verbose:
        print("Average test finished with {:.2f} seconds".format(time.time() - start_time))
        if loss_func is not None:
            print("Loss: {:.6f}".format(test_loss))

    # * Rescale the predicted data
    prediction_list = data.test_output_scaler.inverse_transform(prediction_list)
    return prediction_list


def recurrent_test(model: RNN,
                   data: stock_data,
                   loss_func: torch.nn.modules.loss = None,
                   verbose=False):
    start_time = time.time()

    #* Get information from model
    device = next(model.parameters()).device
    successive_days = model.successive_days

    # * ndarray for storing prediction of machine
    prediction_list = torch.as_tensor(np.zeros((len(data.test_raw) - data.past_len - successive_days + 1, len(data.out_features)), dtype=data.test_raw.dtype), device=device)

    #* Test loader : batch size of 1
    test_loader = data.get_test_loader()

    # * Do prediction
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            #* update input list
            x, y = x.to(device), y.to(device)
            current_successive_days = i % successive_days
            if current_successive_days:
                x[:, -current_successive_days:, data.common_input_idx] = prediction_list[i - current_successive_days: i, data.common_output_idx]

            #* Do prediction
            prediction = model(x)[:, 0, :]

            #* Save the prediction
            prediction_list[i] = prediction.squeeze(0)

            # * Calculate loss
            if loss_func is not None:
                loss = loss_func(prediction, y[:, 0, :])
                test_loss += loss.item()

    #* Average loss
    test_loss /= i + 1

    #* Print the result
    if verbose:
        print("Recurrent test finished with {:.2f} seconds".format(time.time() - start_time))
        if loss_func is not None:
            print("Loss: {:.6f}".format(test_loss))

    # * Rescale the predicted data
    prediction_list = data.test_output_scaler.inverse_transform(prediction_list.cpu().numpy())
    return prediction_list


def plot_prediction(data: stock_data,
                    epoch,
                    avg_prediction: np.ndarray = None,
                    recurrent_prediction: np.ndarray = None,
                    save_path=None):
    num_features = len(data.out_features)
    fig = plt.figure(figsize=(10 * num_features, 10))

    for i in range(num_features):
        ax = plt.subplot(1, num_features, i + 1)
        real_out = data.data_frame[data.val_test_boundary:][data.out_features[i]]
        ax.plot(real_out, 'r-', label='Real')

        #* Plot average prediction
        if avg_prediction is not None:
            ax.plot(real_out.index[:], avg_prediction[:, i], 'b-', label='Average predicted')

        if recurrent_prediction is not None:
            if data.successive_days:
                ax.plot(real_out.index[:-data.successive_days + 1], recurrent_prediction[:, i], 'g-', label='Recurrent predicted')
            else:
                ax.plot(real_out.index, recurrent_prediction[:, i], 'g-', label='Recurrent predicted')

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


def pre_processed_name(stocks, fmt='parquet', comp='snappy'):
    return '.'.join([stocks, fmt, comp])


if __name__ == "main":
    print("This is module utils")
