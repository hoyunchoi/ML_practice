import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm.notebook import tqdm
import torch.autograd.profiler as profiler

from RNN import RNN
from early_stopping import early_stopping
from covid_data import covid_data

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
          early_stop_counter=0,
          early_stop_delta=0,
          min_val_loss=np.Inf,
          use_tqdm=True,
          verbose=1):
    start_time = time.time()

    #* Use the same device where NN is at
    device = next(model.parameters()).device

    #* ndarray for storing loss per each epoch
    train_loss_list = np.zeros(max_epoch)
    val_loss_list = np.zeros(max_epoch)

    #* Whether to use tqdm
    if use_tqdm:
        epoch_range = tqdm(range(max_epoch), desc="Epoch", colour='green')
        trace_func = tqdm.write
    else:
        epoch_range = range(max_epoch)
        trace_func = print

    #* Initialize early stopping object
    if early_stop_counter:
        best_model = None
        best_epoch = -1
        es = early_stopping(verbose=verbose,
                            patience=early_stop_counter,
                            min_val_loss=min_val_loss,
                            delta=early_stop_delta,
                            trace_func=trace_func)

    for epoch in epoch_range:
        #* Training
        model.train()
        train_loss = 0.0
        train_num = 0

        for x, y in train_loader:
            #* Get input, output value
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]

            #* Feed forwarding
            output = model(x)
            loss = loss_func(output, y)

            #* Back propagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            #* Calculate loss
            train_loss += loss.item()
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
                    val_loss += loss.item()
                    val_num += batch_size
            #* End of a epoch. Store average loss of current epoch
            val_loss /= val_num
            val_loss_list[epoch] = val_loss

        #* Print the result
        if verbose > 1:
            trace_func("Epoch {}/{}".format(epoch + 1, max_epoch), end='\t')
            trace_func("train loss: {:.6f}\t validation loss: {:.6f}".format(train_loss, val_loss))

        #* Early stopping
        if early_stop_counter:
            if es(epoch + 1, val_loss):
                best_epoch = epoch
                best_model = model.state_dict()
            if es.early_stop:
                if verbose:
                    trace_func("Early Stopping!")
                break

    if not early_stop_counter:
        best_model = model.state_dict()
        best_epoch = max_epoch - 1

    print("Train finished with {:.6f} seconds".format(time.time() - start_time))
    return train_loss_list, val_loss_list, (best_model, best_epoch)


def plot_loss(train_loss_list: np.ndarray,
              val_loss_list: np.ndarray,
              stop_epoch=None,
              save_path=None):
    assert train_loss_list.shape == val_loss_list.shape, "train loss and validation loss should have same length"

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(np.arange(1, len(train_loss_list) + 1, 1), train_loss_list, 'bo-', label='train')

    #* Plot only when val_loss_list is not zero array
    if np.count_nonzero(val_loss_list):
        ax.plot(np.arange(1, len(val_loss_list) + 1, 1), val_loss_list, 'go-', label='validation')

    if stop_epoch is not None:
        ax.plot([stop_epoch, stop_epoch], [0, max(train_loss_list[0], val_loss_list[0])], 'r--', label="Early Stop")

    ax.set_xlabel("Epoch", fontsize=30)
    ax.set_ylabel("loss", fontsize=30)
    ax.legend(loc='best', fontsize=30, frameon=False)
    ax.tick_params(axis='both', labelsize=20, direction='in')

    if save_path:
        fig.savefig(save_path, facecolor='w')
    else:
        fig.show()
    return ax


def average_test(model: RNN,
                 data: covid_data,
                 loss_func: torch.nn.modules.loss = None,
                 verbose=False):
    start_time = time.time()

    #* Get information from model
    device = next(model.parameters()).device
    successive_days = model.successive_days

    #* ndarray for storing prediction of machine
    prediction_list = np.zeros((len(data.val_raw) - data.past_days, len(data.out_features)))
    prediction_num = np.zeros(len(data.val_raw) - data.past_days)

    #* Test loader : batch size of 1
    test_loader = data.get_val_loader(batch_size=1)

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

    #* Rescale the predicted data
    prediction_list = data.val_output_scaler.inverse_transform(prediction_list)
    return prediction_list, test_loss


def recurrent_test(model: RNN,
                   data: covid_data,
                   loss_func: torch.nn.modules.loss = None,
                   verbose=False):
    start_time = time.time()

    #* Get information from model
    device = next(model.parameters()).device
    successive_days = model.successive_days

    #* ndarray for storing prediction of machine
    prediction_list = torch.as_tensor(np.zeros((len(data.val_raw) - data.past_days - successive_days + 1, len(data.out_features)), dtype=data.val_raw.dtype), device=device)

    #* Test loader : batch size of 1
    test_loader = data.get_val_loader(batch_size=1)

    #* Do prediction
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

            #* Calculate loss
            if loss_func is not None:
                loss = loss_func(prediction, y[:, 0, :])
                test_loss += loss.item() * successive_days

    #* Average loss
    test_loss /= i + 1

    #* Print the result
    if verbose:
        print("Recurrent test finished with {:.2f} seconds".format(time.time() - start_time))
        if loss_func is not None:
            print("Loss: {:.6f}".format(test_loss))

    #* Rescale the predicted data
    prediction_list = data.val_output_scaler.inverse_transform(prediction_list.cpu().numpy())
    return prediction_list, test_loss


def plot_prediction(data: covid_data,
                    epoch: int,
                    avg_prediction: np.ndarray = None,
                    recurrent_prediction: np.ndarray = None,
                    save_path=None):
    num_features = len(data.out_features)
    fig = plt.figure(figsize=(10 * num_features, 10))

    for i in range(num_features):
        ax = plt.subplot(1, num_features, i + 1)
        real_out = data.data_frame[data.train_val_boundary:][data.out_features[i]]
        real_out = real_out[1:]
        ax.plot(real_out, 'r-', label='Real')

        #* Plot average prediction
        if avg_prediction is not None:
            ax.plot(real_out.index[:], avg_prediction[:, i], 'b-', label='Average predicted')

        if recurrent_prediction is not None:
            if data.successive_days > 1:
                ax.plot(real_out.index[:-data.successive_days + 1], recurrent_prediction[:, i], 'g-', label='Recurrent predicted')
            else:
                ax.plot(real_out.index, recurrent_prediction[:, i], 'g-', label='Recurrent predicted')

        ax.set_xlabel("Date", fontsize=25)
        ax.set_ylabel("Cases", fontsize=25)
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

# def plot_confirmed(data:covid_data, prediction:np.ndarray):
#     predicted_confirmed = prediction[:, 0] * prediction[:, 1]
#     confirmed = data.data_frame[data.train_val_boundary:]['confirmed']
#     confirmed = confirmed[1:]
#     fig, ax = plt.subplots(figsize=(10,10))
#     ax.plot(confirmed, 'k-', label='real confirmed')
#     ax.plot(confirmed.index, predicted_confirmed, 'b-', label='predicted confirmed')
#     ax.set_xlabel("Date", fontsize=25)
#     ax.set_ylabel("Cases", fontsize=25)
#     ax.legend(loc='best', fontsize=25, frameon=False)
#     ax.tick_params(axis='both', labelsize=20, direction='in')
#     fig.autofmt_xdate()
#     fig.show()

#     return ax

if __name__ == "__main__":
    print("This is module utils")
