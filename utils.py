import csv
import random
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Removes useless warning when precision, recall or fscore are zero
import warnings

warnings.filterwarnings('ignore', message='(.*)Precision and F-score are ill-defined(.*)')
warnings.filterwarnings('ignore', message='(.*)Recall and F-score are ill-defined(.*)')


class AverageMeterNumPyArray(object):

    def reset(self, size):
        self.np_array = np.zeros(size)
        self.count = np.zeros(size)

    def __init__(self, size):
        self.reset(size)

    def update(self, val, counts):
        self.np_array += val
        self.count += counts

    def average(self):
        return np.divide(self.np_array, self.count, out=np.zeros_like(self.np_array), where=self.count != 0)


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


def calculate_precision_and_recall(outputs, targets, class_idx, func_class_counts):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)

        ground_truth = targets.view(-1, 1).cpu().numpy()
        predictions = pred.cpu().numpy()

        # This is useful to mean the results after the batch completes
        # all_classes = np.append(ground_truth, predictions)
        # count_vector = np.bincount(all_classes)
        # zero_pad = labels_size - len(count_vector)
        # padded_counts = np.pad(count_vector, (0, zero_pad), 'constant')

        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, predictions,
                                                                             labels=class_idx)

        # print(f"prec: {precision} - recall: {recall} - targets: {targets} - pred: {pred}")
        class_counts_for_mean = func_class_counts(support)

        # print(f"prec: {precision}")
        # print(f"rec: {recall}")
        # print(f"fscore: {fscore}")
        # print(f"support: {support}")
        # print(f"class_counts: {class_counts_for_mean}")

        return precision, recall, fscore, class_counts_for_mean


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2 ** 32:
        torch_seed = torch_seed % 2 ** 32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):
    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass
