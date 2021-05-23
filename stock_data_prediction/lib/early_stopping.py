import numpy as np
import torch

"""
    class early_stopping: Early stops the training if validation loss doesn't improve after a given patience.
    Args:
        patience (int): How long to wait after last time validation loss improved.
                        Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
        path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
        trace_func (function): trace print function.
                        Default: print
"""


class early_stopping:
    def __init__(self, patience=7, verbose=False, min_val_loss=np.Inf, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        # self.best_score = None
        self.early_stop = False
        self.min_val_loss = min_val_loss
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, epoch, val_loss, model):
        if val_loss >= self.min_val_loss:
            self.counter += 1
            self.trace_func("Epoch {}\tEarly Stopping counter: {} out of {}".format(epoch, self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._save_checkpoint(epoch, val_loss, model)
            self.counter = 0






        # score = -val_loss

        # if self.best_score is None:
        #     self.best_score = score
        #     self._save_checkpoint(epoch, val_loss, model)
        # elif score < self.best_score + self.delta:
        #     self.counter += 1
        #     self.trace_func("Epoch {}\tEarly Stopping counter: {} out of {}".format(epoch, self.counter, self.patience))
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        # else:
        #     self.best_score = score
        #     self._save_checkpoint(epoch, val_loss, model)
        #     self.counter = 0

    #* Saves model when validation loss decrease.
    def _save_checkpoint(self, epoch, val_loss, model):
        if self.verbose:
            self.trace_func("Epoch {}\tValidation loss decreased ({:.6f} --> {:.6f}). Saving model".format(epoch, self.min_val_loss, val_loss))
        torch.save(model.state_dict(), self.path)
        self.min_val_loss = val_loss
