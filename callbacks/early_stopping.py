import numpy as np
from torch import save


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    WHITE = '\033[97m'

class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation accuracy improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_accuracy_max = -np.Inf  # Change to -Inf for accuracy
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_accuracy, model, epoch):
        score = val_accuracy  # Use accuracy instead of loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= int(0.8 * self.patience):
                self.trace_func(f"Epoch: {epoch}. EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model, epoch):
        """Saves model when validation accuracy increases."""
        if self.verbose:
            increase = (val_accuracy -
                        self.val_accuracy_max) / self.val_accuracy_max * 100 if self.val_accuracy_max > 0 else 0
            increased_txt = (Colors.GREEN + f"({increase:.2f}%)" + Colors.ENDC if increase > 0 else Colors.RED +
                             f"({increase:.2f}%)" + Colors.ENDC)
            self.trace_func(
                f"Epoch: {epoch}. Validation accuracy increased ({self.val_accuracy_max:.6f} --> {val_accuracy:.6f}), {increased_txt}"
                + " Saving model ...")
        save(model.state_dict(), self.path)
        self.val_accuracy_max = val_accuracy  # Update the max accuracy
