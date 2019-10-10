import csv
import random
from functools import partialmethod
from pathlib import Path

import numpy as np
import torch
from torch import device


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def class_counts(x): return 1 if x > 0 else 0


def ground_truth_and_predictions(outputs, targets):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)

        ground_truth = targets.view(-1, 1).cpu().numpy()
        predictions = pred.cpu().numpy()

        return ground_truth, predictions


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2 ** 32:
        torch_seed = torch_seed % 2 ** 32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    # CAVEAT: If using ADAM, the learning rate will be always the same (correctly), and you will get on the plot the same value.
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):
    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, Path):
        path = str(obj)
        return path

    if isinstance(obj, device):
        # For some reason, the ArgumentParser has this type of parameter when dumping the JSON
        return "<not_used_and_not_serializable>"

    return obj.__dict__
