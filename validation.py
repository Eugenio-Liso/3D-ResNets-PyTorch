import time
# Removes useless warning when precision, recall or fscore are zero
import warnings

import torch
from sklearn.metrics import precision_recall_fscore_support

from utils import AverageMeter, calculate_accuracy, ground_truth_and_predictions

warnings.filterwarnings('ignore', message='(.*)Precision and F-score are ill-defined(.*)')
warnings.filterwarnings('ignore', message='(.*)Recall and F-score are ill-defined(.*)')


def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              class_names,
              logger,
              tb_writer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    class_size = len(class_names)
    class_idx = list(range(0, class_size))

    ground_truth_labels = []
    predicted_labels = []

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            ground_truth, predictions = ground_truth_and_predictions(outputs, targets)

            ground_truth_labels.extend(ground_truth)
            predicted_labels.extend(predictions)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies))

            end_time = time.time()

    losses_avg = losses.avg
    accuracies_avg = accuracies.avg

    precision_epoch, recall_epoch, fscore_epoch, _ = \
        precision_recall_fscore_support(ground_truth_labels,
                                        predicted_labels,
                                        labels=class_idx)

    # print("update")
    # print(f"prec avg:{precs_avg}")
    # print(f"recs_avg: {recs_avg}")
    # print(f"fscores_avg: {fscores_avg}")

    logger.log({
        'epoch': epoch,
        'loss': losses_avg,
        'acc': accuracies_avg,
        'prec': precision_epoch,
        'rec': recall_epoch,
        'fscore': fscore_epoch
    })

    if tb_writer is not None:
        tb_writer.add_scalar('Validation/Loss per epoch', losses_avg, epoch)
        tb_writer.add_scalar('Validation/Accuracy per epoch', accuracies_avg, epoch)

        [tb_writer.add_scalar(f"Validation/Precision for class {class_names[idx]} per epoch", precision_epoch[idx],
                              epoch) for
         idx in class_idx]
        [tb_writer.add_scalar(f"Validation/Recall for class {class_names[idx]} per epoch", recall_epoch[idx], epoch) for
         idx
         in class_idx]
        [tb_writer.add_scalar(f"Validation/F-Score for class {class_names[idx]} per epoch", fscore_epoch[idx], epoch)
         for
         idx in class_idx]

    return losses.avg
