import torch
import time
import sys

from utils import AverageMeter, AverageMeterNumPyArray, calculate_accuracy, calculate_precision_and_recall, class_counts
import numpy as np

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
    precs = AverageMeterNumPyArray(class_size)
    recs = AverageMeterNumPyArray(class_size)
    fscores = AverageMeterNumPyArray(class_size)
    class_idx = list(range(0, class_size))
    func_class_counts = np.vectorize(class_counts)

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            prec, rec, fscore, counts = calculate_precision_and_recall(outputs, targets, class_idx, func_class_counts)

            precs.update(prec, counts)
            recs.update(rec, counts)
            fscores.update(fscore, counts)

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

    precs_avg = precs.average()
    recs_avg = recs.average()
    fscores_avg = fscores.average()

    # print("update")
    # print(f"prec avg:{precs_avg}")
    # print(f"recs_avg: {recs_avg}")
    # print(f"fscores_avg: {fscores_avg}")

    logger.log({
        'epoch': epoch,
        'loss': losses_avg,
        'acc': accuracies_avg,
        'prec': precs_avg,
        'rec': recs_avg,
        'fscore': fscores_avg
    })

    if tb_writer is not None:
        tb_writer.add_scalar('Validation/Loss per epoch', losses_avg, epoch)
        tb_writer.add_scalar('Validation/Accuracy per epoch', accuracies_avg, epoch)

        [tb_writer.add_scalar(f"Validation/Precision for class {class_names[idx]} per epoch", precs_avg[idx], epoch) for idx in class_idx]
        [tb_writer.add_scalar(f"Validation/Recall for class {class_names[idx]} per epoch", recs_avg[idx], epoch) for idx in class_idx]
        [tb_writer.add_scalar(f"Validation/F-Score for class {class_names[idx]} per epoch", fscores_avg[idx], epoch) for idx in class_idx]

    return losses.avg