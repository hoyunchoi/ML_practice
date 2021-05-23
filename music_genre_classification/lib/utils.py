import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from tqdm.notebook import tqdm

from genre_data import genre_data
from early_stopping import early_stopping
"""
    Various functions for making stock price predicting machine

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
    plot_loss_accuracy(train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list, save_path=None):
    Args:
        train_loss_list : list of training loss
        val_loss_list : list of validation loss
        train_accuracy_list : list of training accuracy
        val_accuracy_list : list of validation accuracy
        save_path : If given, save the plot to the save_path
    ---------------------------------------------------------------------
"""


def model_summary(model: torch.nn, input_size, precision='64'):
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


def train(model: torch.nn,
          max_epoch,
          loss_func: torch.nn.modules.loss,
          optimizer: torch.optim,
          train_loader: torch.utils.data.dataloader.DataLoader,
          val_loader: torch.utils.data.dataloader.DataLoader = None,
          early_stop_counter=0,
          min_val_loss=np.Inf,
          verbose=1):

    start_time = time.time()

    #* Use the same device where NN is at
    device = next(model.parameters()).device

    #* ndarray for storing loss and accuracy per each epoch
    train_loss_list, train_accuracy_list = np.zeros(max_epoch), np.zeros(max_epoch)
    val_loss_list, val_accuracy_list = np.zeros(max_epoch), np.zeros(max_epoch)

    #* Initialize early stopping object
    if early_stop_counter:
        best_model = None
        best_epoch = -1
        es = early_stopping(verbose=verbose,
                            patience=early_stop_counter,
                            min_val_loss=min_val_loss,
                            delta=1e-3,
                            trace_func=tqdm.write)

    for epoch in tqdm(range(max_epoch), desc="Epoch", colour='green'):
        #* Training
        model.train()
        train_loss = 0.0
        train_num = 0
        train_accuracy = 0.0

        for x, y in train_loader:
            #* Get input, output value
            x, y = x.to(device), y.to(device)
            batch_size = x.shape[0]

            #* Feed forwarding
            output = model(x)
            prediction = torch.argmax(output, dim=1)
            loss = loss_func(output, y)

            #* Back propagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            #* Calculate loss and accuracy
            train_loss += loss.item()
            train_num += batch_size
            train_accuracy += (prediction == y).sum()

        #* End of a epoch. Store average loss, accuracy of current epoch
        train_loss /= train_num
        train_accuracy /= train_num / 100.0
        train_loss_list[epoch] = train_loss
        train_accuracy_list[epoch] = train_accuracy

        #* Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_num = 0
            val_accuracy = 0.0

            with torch.no_grad():
                for x, y in val_loader:
                    #* Get input, output value
                    x, y = x.to(device), y.to(device)
                    batch_size = x.shape[0]

                    #* Feed forwarding
                    output = model(x)
                    prediction = torch.argmax(output, dim=1)
                    loss = loss_func(output, y)

                    #* Calculate loss and accuracy
                    val_loss += loss.item()
                    val_num += batch_size
                    val_accuracy += (prediction == y).sum()

            #* End of a epoch. Store average loss, accuracy of current epoch
            val_loss /= val_num
            val_accuracy /= val_num / 100.0
            val_loss_list[epoch] = val_loss
            val_accuracy_list[epoch] = val_accuracy

        #* Print the result
        if verbose > 1:
            tqdm.write("Epoch {}/{}".format(epoch + 1, max_epoch), end='\t')
            tqdm.write("train accuracy(loss): {:.1f}%({:.6f})".format(train_accuracy, train_loss), end='\t')
            tqdm.write("validation accuracy(loss): {:.1f}%({:.6f})".format(val_accuracy, val_loss))

        #* Early stopping
        if early_stop_counter:
            if es(epoch + 1, val_loss):
                best_epoch = epoch
                best_model = model.state_dict()
            if es.early_stop:
                tqdm.write("Early Stopping!")
                break

    if not early_stop_counter:
        best_model = model.state_dict()
        best_epoch = max_epoch - 1

    print("Train finished with {:.6f} seconds".format(time.time() - start_time))
    return (train_loss_list, val_loss_list), (train_accuracy_list, val_accuracy_list), (best_model, best_epoch)


def plot_loss_accuracy(train_loss_list: np.ndarray,
                       val_loss_list: np.ndarray,
                       train_accuracy_list: np.ndarray,
                       val_accuracy_list: np.ndarray,
                       stop_epoch=None,
                       save_path=None):
    assert train_loss_list.shape == val_loss_list.shape, "train loss and validation loss should have same length"
    assert train_accuracy_list.shape == val_accuracy_list.shape, "train accuracy and validation accuracy should have same length"

    fig = plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(np.arange(1, len(train_loss_list) + 1, 1), train_loss_list, 'bo-', label='train')
    #* Plot only when val_loss_list is not zero array
    if np.count_nonzero(val_loss_list):
        ax1.plot(np.arange(1, len(val_loss_list) + 1, 1), val_loss_list, 'go-', label='validation')

    if stop_epoch is not None:
        ax1.plot([stop_epoch, stop_epoch], [0, train_loss_list[0]], 'r--', label="Early Stop")

    ax1.set_xlabel('epoch', fontsize=30)
    ax1.set_ylabel('loss', fontsize=30)
    ax1.legend(loc='best', fontsize=30, frameon=False)
    ax1.tick_params(axis='both', labelsize=20, direction='in')

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(np.arange(1, len(train_loss_list) + 1, 1), train_accuracy_list, 'bo-', label='train')
    #* Plot only when val_accuracy_list is not zero array
    if np.count_nonzero(val_accuracy_list):
        ax2.plot(np.arange(1, len(val_loss_list) + 1, 1), val_accuracy_list, 'go-', label='validation')

    if stop_epoch is not None:
        ax2.plot([stop_epoch, stop_epoch], [train_accuracy_list[0], train_accuracy_list[-1]], 'r--', label="Early Stop")

    ax2.set_xlabel('epoch', fontsize=30)
    ax2.set_ylabel('accuracy(%)', fontsize=30)
    ax2.legend(loc='best', fontsize=30, frameon=False)
    ax2.tick_params(axis='both', labelsize=20, direction='in')

    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()
    return ax1, ax2


def test(model: torch.nn,
         data: genre_data,
         loss_func: torch.nn.modules.loss = None,
         verbose=False):
    start_time = time.time()

    #* Use the same device where NN is at
    device = next(model.parameters()).device

    #* Prepare for testing
    test_loader = data.get_test_loader()
    test_tot = {label: 0 for label in np.arange(len(data.genres))}
    test_correct = {label: 0 for label in np.arange(len(data.genres))}

    #* Do prediction
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            #* Do prediction
            x,y = x.to(device), y.to(device)
            output = model(x)
            prediction = torch.argmax(output, dim=1)
            label = y.tolist()[0]

            #* Check accuracy
            test_tot[label] += 1
            if prediction == y:
                test_correct[label] += 1

            #* Calculate loss
            if loss_func is not None:
                loss = loss_func(output, y)
                test_loss += loss.item()

    #* Store the loss and prediction per each genre
    test_loss /= i + 1
    accuracy = {}
    for (label, tot), correct in zip(test_tot.items(), test_correct.values()):
        genre = data.genres[label]
        accuracy[genre] = correct / tot * 100
    accuracy['tot'] = sum(test_correct.values()) / sum(test_tot.values()) * 100

    if verbose:
        if loss_func is not None:
            print("Test loss: {:.6f}".format(test_loss), end='\t')
        print("Test accuracy: {:.2f}%".format(accuracy['tot']))
    print("Test finished with {:.2f} seconds".format(time.time() - start_time))

    return accuracy, (test_loss, accuracy)


def plot_accuracy(accuracy: np.ndarray,
                  epoch = 0,
                  save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    bar_list = ax.bar(list(accuracy.keys()), accuracy.values(), color='slateblue', width=0.5)
    bar_list[-1].set_color('darkslateblue')
    for patch in ax.patches:
        left, _, width, height = patch.get_bbox().bounds
        ax.annotate("{:.1f}%".format(height), (left + width / 2, height * 1.01), ha='center', fontsize=13)
    ax.set_xlabel("Genre", fontsize=25)
    ax.set_ylabel("Accuracy", fontsize=25)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_title("Test result after {} epochs".format(epoch), fontsize=35)

    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()
    return ax
